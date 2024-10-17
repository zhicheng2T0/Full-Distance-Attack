import torch
import torch.fft as fft

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as T
import torchvision

import time
import os
import PIL

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F


def rgb2hsv_torch(img):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + 0.0001 ) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + 0.0001 )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)
    return hsv

def hsv2rgb_torch(hsv):
    h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]

    h = h%1
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb




class MappingNet(nn.Module):
    #a network trained to correct color shifts caused by the printer used
    def __init__(self,width):
        super(MappingNet,self).__init__()

        self.fc1 = nn.Linear(3, width)
        self.bn1 = torch.nn.BatchNorm1d(width)


        self.bn1 = torch.nn.BatchNorm1d(width)

        self.block1 = nn.Sequential(
                        nn.Linear(width, width),
                        nn.ReLU(),
                        )

        self.fcout = nn.Linear(width, 3)
        self.out_sig=nn.Sigmoid()

    def forward(self,x):
        x_ori=x
        x=self.fc1(x)
        x=self.bn1(x)
        x=x+self.block1(x)
        x=self.fcout(x)
        x=self.out_sig(x)*2-1+x_ori
        x=torch.clamp(x,0,1)
        return x


class BlurModule:
    #module for the atmospheric perspective module and the camera simulation module
    def __init__(self, strides, widths, temperatures,widths2,temperatures2, N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity,  sv_shift, device, trainable=True,load=True, model_name='BlurModule_default',num_days=1):

        self.variable_list=[]
        load_index=0

        self.trainable=trainable
        self.device=device
        self.load=load
        self.model_name=model_name

        self.sigmoid_func=torch.nn.Sigmoid()

        self.strides=strides
        self.kernel_sizes=[]
        for i in range(len(self.strides)):
            cur_ksize=int(self.strides[i]+2)
            if cur_ksize<3:
                cur_ksize=3
            elif cur_ksize%3!=0:
                cur_ksize=int(cur_ksize+1)
            self.kernel_sizes.append(cur_ksize)
        self.widths=widths
        self.temperatures=temperatures
        self.widths2=widths2
        self.temperatures2=temperatures2
        self.height=1
        self.channels=3

        self.widths=torch.tensor(widths).to(device)
        self.widths,load_index,self.variable_list=self.make_variable_differentiable(self.widths,self.load,load_index,self.variable_list)
        self.temperatures=torch.tensor(temperatures).to(device)

        self.widths2=torch.tensor(widths2).to(device)
        self.widths2,load_index,self.variable_list=self.make_variable_differentiable(self.widths2,self.load,load_index,self.variable_list)
        self.temperatures2=torch.tensor(temperatures2).to(device)

        self.N_exp=torch.tensor(N_exp).to(device)
        self.N_exp,load_index,self.variable_list=self.make_variable_differentiable(self.N_exp,self.load,load_index,self.variable_list)
        self.c_exp=torch.tensor(c_exp).to(device)
        self.c_exp,load_index,self.variable_list=self.make_variable_differentiable(self.c_exp,self.load,load_index,self.variable_list)
        self.lambda_divide=torch.tensor(lambda_divide).to(device)
        self.lambda_divide,load_index,self.variable_list=self.make_variable_differentiable(self.lambda_divide,self.load,load_index,self.variable_list)
        self.rbg_lambda=torch.tensor(rbg_lambda).to(device)
        self.rbg_lambda,load_index,self.variable_list=self.make_variable_differentiable(self.rbg_lambda,self.load,load_index,self.variable_list)
        self.sky_rbg=torch.tensor(sky_rbg).to(device)
        temp_turb=[]
        for i in range(num_days):
            temp_turb.append(turbidity)
        self.turbidity=torch.tensor(temp_turb).to(device)
        self.turbidity,load_index,self.variable_list=self.make_variable_differentiable(self.turbidity,self.load,load_index,self.variable_list)
        sv_temp=[]
        for i in range(num_days):
            sv_temp.append(sv_shift)
        self.sv_shift=torch.tensor(sv_temp).to(device)
        self.sv_shift,load_index,self.variable_list=self.make_variable_differentiable(self.sv_shift,self.load,load_index,self.variable_list)
        self.sv_shift0=torch.tensor([0,0]).to(device)

        self.blur_base=[]
        self.blur_base_v4=[]
        for k in range(len(self.kernel_sizes)):
            cur_blur_base=[]
            cur_blur_base_v4=[]
            for i in range(int(self.kernel_sizes[k])):
                cur=[]
                for j in range(int(self.kernel_sizes[k])):
                    cur.append([i,j])
                cur_blur_base.append(cur)
                cur_blur_base_v4.append(cur)
            cur_blur_base=torch.tensor(np.asarray(cur_blur_base)).to(self.device)
            self.blur_base.append(cur_blur_base)
            self.blur_base_v4.append(cur_blur_base)


        self.EOT_temp=1
        self.EOT_N_exp=0.4
        self.EOT_c_exp=0.4
        self.EOT_lambda_divide=2
        self.EOT_rbg_lambda=10
        self.EOT_sky_rbg=0.05
        self.EOT_turbidity=0.3
        self.EOT_blur_width=[0.1,0.9]
        self.EOT_blur_width2=[0.1,0.9]

        self.dist_list=[4,8,14,20,26,34,42]


    def make_variable_differentiable(self,variable,load,load_index,variable_list):
        if load==True:
            value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
            variable=value
            load_index+=1
        variable = torch.tensor(variable).float()
        variable = variable.to(self.device)
        if self.trainable==True:
            variable.requires_grad_(True)
        variable_list.append(variable)
        return variable,load_index,variable_list


    def save(self,epoch):
        save_dir='./model/'+self.model_name
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
        for i in range(len(self.variable_list)):
            cur_name=save_dir+'/'+str(epoch)+'_'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())
            cur_name=save_dir+'/'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())

    def get_adjustable_kernelv4(self,config,counter):
        '''
        k_size,channels,temperature,height,width
        Note that temperature, width are tensors and should be differentiable
        '''
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        height=config[4]
        width=config[1]

        temperature=torch.clamp(temperature,0,100)
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=self.blur_base_v4[counter]
        cur_x=result[:,:,0:1]+x_start
        cur_y=result[:,:,1:2]+y_start
        cur_x_abs=torch.abs(cur_x)
        cur_y_abs=torch.abs(cur_y)
        cur_xy=torch.cat([cur_x_abs,cur_y_abs],2)
        cur_sigx,_=torch.max(cur_xy,2)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)


        return result

    def get_blurring_kernel_v4(self,kernel_size,width,temp,stride,height,channels,counter):
        blur_kernel=self.get_adjustable_kernelv4([kernel_size,width,temp,stride,height,channels],counter)

        return blur_kernel

    def get_adjustable_kernel(self,config,counter):
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        temperature=torch.clamp(temperature,0,100)
        height=config[4]
        width=config[1]
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=self.blur_base[counter]
        cur_x=result[:,:,0]+x_start
        cur_y=result[:,:,1]+y_start
        cur_sigx=torch.sqrt(cur_x*cur_x+cur_y*cur_y)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)


        return result

    def get_blurring_kernel(self,kernel_size,width,temp,stride,height,channels,counter):
        blur_kernel=self.get_adjustable_kernel([kernel_size,width,temp,stride,height,channels],counter)

        return blur_kernel

    def run_kernel(self,input,kernel,stride,type):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):
            decision_value=(in_s[2+i]-1)%stride
            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])
            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride
                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]

        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)
        return filtered

    def calculate_gaussian(self,x,hue_mean,hue_std):
        cur_mean=torch.clamp(hue_mean,0,1)
        cur_mean=torch.reshape(cur_mean,[1,cur_mean.shape[0],1,1])
        cur_mean=cur_mean.repeat(1,1,x.shape[2],x.shape[3])
        diffs=(x-cur_mean)*(x-cur_mean)

        cur_std=torch.clamp(hue_std,0,1)
        cur_std=torch.reshape(cur_std,[1,cur_std.shape[0],1,1])
        cur_std=cur_std.repeat(1,1,x.shape[2],x.shape[3])
        cur_std=cur_std+0.001
        cur_std_square=cur_std*cur_std

        results=(1/(cur_std_square*2*3.14))*torch.exp((-0.5)*(diffs/cur_std_square))
        results=torch.mean(results,dim=1,keepdims=True)
        return results

    def get_exp_value(self,T,Lambda,s,h_0=1,eot_size=0):
        n=1.0003

        change_val=self.EOT_N_exp*np.random.rand()-self.EOT_N_exp/2
        N_exp_temp=self.N_exp+change_val*eot_size
        N_exp_temp=torch.clamp(N_exp_temp,0.1,100)

        N=2.545*(10**N_exp_temp)
        p_n=0.035
        H_r0=7994
        K_R=1.0396

        change_val=self.EOT_lambda_divide*np.random.rand()-self.EOT_lambda_divide/2
        lambda_divide_temp=self.lambda_divide+change_val*eot_size
        lambda_divide_temp=torch.clamp(lambda_divide_temp,0.1,100)
        Lambda_cur=Lambda/(lambda_divide_temp*1000)

        beta_r=(8*(np.pi**3)*((n*n-1)**2))/(3*N*(Lambda_cur**4))*((6+3*p_n)/(6-7*p_n))*np.exp(-1*(h_0/H_r0))*K_R


        change_val=self.EOT_c_exp*np.random.rand()-self.EOT_c_exp/2
        c_exp_temp=self.c_exp+change_val*eot_size
        c_exp_temp=torch.clamp(c_exp_temp,0.1,100)
        if T=='self':
            c=(0.6544*self.turbidity-0.6510)*(1/(10**c_exp_temp))
        else:
            c=(0.6544*T-0.6510)*(1/(10**c_exp_temp))
        K_M=0.0092
        v=4
        H_m0=1200

        beta_m=0.434*c*np.pi*((2*np.pi/Lambda_cur)**(v-2))*np.exp(-1*h_0/H_m0)*K_M

        result=torch.exp(-1*(beta_r+beta_m)*s)

        result=torch.clamp(result,0,1)

        return result

    def run_blurring(self,input,counter,turbidity='self', sky_rbg='self', eot_size=0, useblur=True, day_index='default',blur_eot_factor=15):

        filtered=input

        if eot_size==0:
            cur_width=self.widths*self.strides[counter]
        else:
            cur_width=self.widths*self.strides[counter]+eot_size*blur_eot_factor*(np.random.rand()-0.5)
            cur_width=torch.clamp(cur_width,self.EOT_blur_width[0]*self.strides[counter],self.EOT_blur_width[1]*self.strides[counter])

        cur_temp=self.temperatures[counter]

        blur_kernel=self.get_blurring_kernel(self.kernel_sizes[counter],cur_width,cur_temp,1,self.height,self.channels,counter)

        if eot_size==0:
            cur_width2=self.widths2*self.strides[counter]
        else:
            cur_width2=self.widths2*self.strides[counter]+eot_size*blur_eot_factor*(np.random.rand()-0.5)
            cur_width2=torch.clamp(cur_width2,self.EOT_blur_width2[0]*self.strides[counter],self.EOT_blur_width2[1]*self.strides[counter])

        cur_temp2=self.temperatures2

        cur_ksize2=int(self.strides[counter]+2)
        if cur_ksize2<3:
            cur_ksize2=3
        elif cur_ksize2%3!=0:
            cur_ksize2=int(cur_ksize2+1)

        blur_kernel2=self.get_blurring_kernel_v4(self.kernel_sizes[counter],cur_width2,cur_temp2,self.strides[counter],self.height,self.channels,counter)

        change_val=self.EOT_rbg_lambda*np.random.rand()-self.EOT_rbg_lambda/2
        rbg_lambda_temp=self.rbg_lambda+change_val*eot_size
        rbg_lambda_temp=torch.clamp(rbg_lambda_temp,38,78)
        change_val=self.EOT_turbidity*np.random.rand()-self.EOT_turbidity/2
        if turbidity=='self':
            turbidity_temp=self.turbidity[day_index]+change_val*eot_size
        else:
            turbidity_temp=turbidity+change_val*eot_size
        turbidity_temp=torch.clamp(turbidity_temp,1,20)

        cur_rgbl=torch.reshape((rbg_lambda_temp*10),[1,3,1,1])
        cur_exp=self.get_exp_value(turbidity_temp,cur_rgbl,self.dist_list[counter],eot_size=eot_size)
        cur_exp=torch.Tensor.repeat(cur_exp,[1,1,filtered.shape[2],filtered.shape[3]])

        if sky_rbg=='self':
            change_val=self.EOT_sky_rbg*np.random.rand()-self.EOT_sky_rbg/2
            sky_rbg_temp=self.sky_rbg+change_val*eot_size
            sky_rbg_temp=torch.clamp(sky_rbg_temp,0,1)
            cur_sky_rbg=torch.reshape(sky_rbg_temp,[1,3,1,1])
        else:
            change_val=self.EOT_sky_rbg*np.random.rand()-self.EOT_sky_rbg/2
            sky_rbg_temp=sky_rbg+change_val*eot_size
            sky_rbg_temp=torch.clamp(sky_rbg_temp,0,1)
            cur_sky_rbg=torch.reshape(sky_rbg_temp,[1,3,1,1])
        cur_sky_rbg=torch.Tensor.repeat(cur_sky_rbg,[1,1,filtered.shape[2],filtered.shape[3]])

        filtered=filtered*cur_exp+cur_sky_rbg*(1-cur_exp)

        filtered_hsv=rgb2hsv_torch(filtered)
        filtered_h=filtered_hsv[:,0:1,:,:]
        filtered_s=filtered_hsv[:,1:2,:,:]
        filtered_v=filtered_hsv[:,2:3,:,:]
        if day_index=='default':
            filtered_s=filtered_s+self.sv_shift0[0]
            filtered_v=filtered_v+self.sv_shift0[1]
        else:
            filtered_s=filtered_s+self.sv_shift[day_index][0]
            filtered_v=filtered_v+self.sv_shift[day_index][1]
        filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
        filtered_hsv=torch.clamp(filtered_hsv,0,1)
        filtered=hsv2rgb_torch(filtered_hsv)
        filtered=torch.clamp(filtered,0,1)

        if useblur==True:
            cur_resize = T.Resize(size = [filtered.shape[2],filtered.shape[3]])
            filtered=self.run_kernel(filtered,blur_kernel,1,'expand')
            filtered=cur_resize(filtered)
            filtered=self.run_kernel(filtered,blur_kernel2,self.strides[counter],'same')
            filtered=torch.clamp(filtered,0,1)
        else:
            filtered=input
            filtered=torch.clamp(filtered,0,1)


        return filtered




class StyleFilters:
    #module for the style filter simulation module
    def __init__(self, sharpen_configs, desharpen_configs, device, contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name='stylefilters_default',load=False,manual_keep=None,trainable=True):

        '''
        Effects to be added
        1. contrast
        2. vibrance
        3. highlight and shadows
        '''


        self.trainable=trainable

        self.manual_keep=manual_keep
        load_index=0
        self.device=device

        self.model_name=model_name

        self.variable_list=[]

        self.sharpen_configs=sharpen_configs

        self.desharpen_configs,self.desharpen_diff_indexes,load_index=self.preprocess_configs(desharpen_configs,self.variable_list,load,load_index)

        self.sigmoid_func=torch.nn.Sigmoid()

        self.desharpen_base=[]
        for i in range(int(self.desharpen_configs[0])):
            cur=[]
            for j in range(int(self.desharpen_configs[0])):
                cur.append([i,j])
            self.desharpen_base.append(cur)
        self.desharpen_base=torch.tensor(np.asarray(self.desharpen_base)).to(self.device)


        self.contrast_val=torch.tensor(contrast_val).to(device)
        self.contrast_val,load_index,self.variable_list=self.make_variable_differentiable(self.contrast_val,load,load_index,self.variable_list)

        self.vibrance_val=torch.tensor(vibrance_val).to(device)
        self.vibrance_val,load_index,self.variable_list=self.make_variable_differentiable(self.vibrance_val,load,load_index,self.variable_list)

        self.shahigh_val=torch.tensor(np.asarray(shahigh_val)).to(device)
        self.shahigh_val,load_index,self.variable_list=self.make_variable_differentiable(self.shahigh_val,load,load_index,self.variable_list)

        self.exposure_val=torch.tensor(exposure_val).to(device)
        self.exposure_val,load_index,self.variable_list=self.make_variable_differentiable(self.exposure_val,load,load_index,self.variable_list)

        self.color_temp_r_b=torch.tensor(color_temp_r_b).to(device)
        self.color_temp_r_b,load_index,self.variable_list=self.make_variable_differentiable(self.color_temp_r_b,load,load_index,self.variable_list)

        self.EOT_ds_width=0.3
        self.EOT_ds_temp=1
        self.EOT_ob_width=0.8
        self.EOT_ob_temp=1
        self.EOT_contrast=0.08
        self.EOT_vibrance=0.15
        self.EOT_shahigh=[[0.15,0.15]]
        self.EOT_exposure=0.1
        self.EOT_color_temp_r_b=0.03
        self.EOT_gmm_prob=0.5

    def make_variable_differentiable(self,variable,load,load_index,variable_list):
        if load==True:
            value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
            variable=value
            load_index+=1
        variable = torch.tensor(variable).float()
        variable = variable.to(self.device)
        if self.trainable==True:
            variable.requires_grad_(True)
        variable_list.append(variable)
        return variable,load_index,variable_list

    def save(self,epoch):
        save_dir='./model/'+self.model_name
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
        for i in range(len(self.variable_list)):
            cur_name=save_dir+'/'+str(epoch)+'_'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())
            cur_name=save_dir+'/'+str(i)+'.npy'
            if self.device=='cpu':
                np.save(cur_name,self.variable_list[i].clone().detach().numpy())
            else:
                np.save(cur_name,self.variable_list[i].cpu().clone().detach().numpy())

    def preprocess_configs(self,configs,all_variables,load,load_index):
        blur_configs=configs[0]
        blur_diff_indexes=configs[1]
        for i in range(len(blur_diff_indexes)):
            if load==True:
                value=np.load('./model/'+self.model_name+'/'+str(load_index)+'.npy')
                blur_configs[blur_diff_indexes[i]]=value
                load_index+=1
            blur_configs[blur_diff_indexes[i]] = torch.tensor(blur_configs[blur_diff_indexes[i]]).float()
            blur_configs[blur_diff_indexes[i]] = blur_configs[blur_diff_indexes[i]].to(self.device)
            if self.trainable==True:
                blur_configs[blur_diff_indexes[i]].requires_grad_(True)
            all_variables.append(blur_configs[blur_diff_indexes[i]])
        return blur_configs,blur_diff_indexes,load_index


    def get_adjustable_kernel(self,config,base):
        k_size=config[0]
        channels=config[5]
        temperature=config[2]
        temperature=torch.clamp(temperature,0,100)
        height=config[4]
        width=config[1]
        width=torch.clamp(width,0.1,100)

        x_start=-k_size//2
        x_end=k_size//2
        y_start=-k_size//2
        y_end=k_size//2

        result=base
        cur_x=result[:,:,0]+x_start
        cur_y=result[:,:,1]+y_start
        cur_sigx=torch.sqrt(cur_x*cur_x+cur_y*cur_y)
        forward_x=cur_sigx+(width/2)
        backward_x=-1*cur_sigx+(width/2)
        forward_y=self.sigmoid_func(forward_x*temperature)
        backward_y=self.sigmoid_func(backward_x*temperature)
        result=height*(forward_y+backward_y-1)

        result=torch.unsqueeze(result,0)
        result=torch.unsqueeze(result,0)
        result=result/torch.sum(result)
        result=result.repeat(channels,1,1,1)

        return result


    def run_filter_adjustable(self,input,kernel,stride,type='normal'):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):

            decision_value=(in_s[2+i]-1)%stride

            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])

            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)

                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride

                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]


        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)

        return filtered

    def get_sharpen_kernel(self,sharp_type,sharp_k_size):
        middle=sharp_k_size//2
        if sharp_type=='full':
            center_val=sharp_k_size*sharp_k_size

            kernel=-1*np.ones([3,1,sharp_k_size,sharp_k_size])
            for i in range(3):
                kernel[i,0,middle,middle]=center_val

        elif sharp_type=='cross':
            center_val=(sharp_k_size*2-1)

            kernel=np.zeros([3,1,sharp_k_size,sharp_k_size])
            for i in range(3):
                for j in range(sharp_k_size):
                    kernel[i,0,middle,j]=-1
                    kernel[i,0,j,middle]=-1
                    if j==middle:
                        kernel[i,0,j,j]=center_val

        kernel=torch.tensor(kernel).float().to(self.device)
        return kernel





    def get_blur_sharp(self,sharpen_configs,desharpen_configs):
        sharp_kernel=self.get_sharpen_kernel(sharpen_configs[0],sharpen_configs[1])
        desharpen_kernel=self.get_adjustable_kernel(desharpen_configs,self.desharpen_base)

        return sharp_kernel,desharpen_kernel

    def run_kernel(self,input,kernel,stride,type):
        in_s=input.shape
        ker_s=kernel.shape

        pad_crop_config=[]

        for i in range(2):
            decision_value=(in_s[2+i]-1)%stride
            if decision_value==0:
                pad_num=ker_s[2+i]//2
                if type=='expand':
                    pad_num=pad_num+stride
                remove1=False
                pad_crop_config.append([pad_num,remove1])
            else:
                if decision_value%2!=0:
                    remove1=True
                    pad_num=(ker_s[2+i]//2)-((decision_value-1)//2)
                else:
                    remove1=False
                    pad_num=(ker_s[2+i]//2)-((decision_value)//2)
                    if type=='expand':
                        pad_num=pad_num+stride
                pad_crop_config.append([pad_num,remove1])

        for i in range(len(pad_crop_config)):
            if i==0 and pad_crop_config[i][1]==True:
                input=input[:,:,0:in_s[2]-1,:]
            elif i==1 and pad_crop_config[i][1]==True:
                input=input[:,:,:,0:in_s[3]-1]

        filtered=torch.nn.functional.conv2d(input=input,weight=kernel,stride=stride,padding=[pad_crop_config[0][0],pad_crop_config[1][0]],groups=3)
        return filtered


    def run_sharpen(self,input,eot_size=0):
        desharpen_configs_temp=self.desharpen_configs.copy()

        change_val=self.EOT_ds_width*np.random.rand()-self.EOT_ds_width/2
        desharpen_configs_temp[1]=self.desharpen_configs[1]+change_val*eot_size

        change_val=self.EOT_ds_temp*np.random.rand()-self.EOT_ds_temp/2
        desharpen_configs_temp[2]=self.desharpen_configs[2]+change_val*eot_size


        kernels=self.get_blur_sharp(self.sharpen_configs,
                            desharpen_configs_temp)

        filtered=self.run_kernel(input,kernels[0],1,'normal')
        filtered=self.run_filter_adjustable(filtered,kernels[1],self.desharpen_configs[3])
        filtered=torch.clamp(filtered,0,1)

        return filtered

    def run_manual_crop(self,input,counter):
        result=input[:,:,self.manual_keep[counter][0][0]:input.shape[2]-self.manual_keep[counter][0][1],self.manual_keep[counter][1][0]:input.shape[3]-self.manual_keep[counter][1][1]]
        return result


    def run_style_filter(self,input,eot_size=0):

        change_val=self.EOT_contrast*np.random.rand()-self.EOT_contrast/2
        contrast_val_temp=self.contrast_val+change_val*eot_size

        filtered=torch.clamp(input,0,1)
        filtered=filtered*255
        cur_cv=torch.clamp(contrast_val_temp,-1,0.3)
        factor = (259 * (cur_cv*255 + 255)) / (255 * (259 - cur_cv*255))
        Red=filtered[:,0:1,:,:]
        Green=filtered[:,1:2,:,:]
        Blue=filtered[:,2:3,:,:]
        newRed   = torch.clamp((factor * (Red - 128) + 128),0,255)
        newGreen = torch.clamp((factor * (Green - 128) + 128),0,255)
        newBlue  = torch.clamp((factor * (Blue  - 128) + 128),0,255)
        filtered=torch.cat([newRed,newGreen,newBlue],1)
        filtered=filtered/255
        filtered=torch.clamp(filtered,0,1)

        shahigh_val_temp=self.shahigh_val.clone()

        change_val=self.EOT_shahigh[0][0]*np.random.rand()-self.EOT_shahigh[0][0]/2
        shahigh_val_temp[0][0]=self.shahigh_val[0][0]+change_val*eot_size

        change_val=self.EOT_shahigh[0][1]*np.random.rand()-self.EOT_shahigh[0][1]/2
        shahigh_val_temp[0][1]=self.shahigh_val[0][1]+change_val*eot_size

        clipped_shahigh_val=torch.clamp(shahigh_val_temp,0,1)
        differenece_shahigh=torch.clamp((clipped_shahigh_val[0][1]-clipped_shahigh_val[0][0]),0.001,1)
        filtered=(filtered-clipped_shahigh_val[0][0])/differenece_shahigh*(1-0)+0
        filtered=torch.clamp(filtered,0,1)

        change_val=self.EOT_exposure*np.random.rand()-self.EOT_exposure/2
        exposure_val_temp=self.exposure_val+change_val*eot_size

        filtered=filtered*(2**exposure_val_temp)
        filtered=torch.clamp(filtered,0,1)

        change_val=self.EOT_vibrance*np.random.rand()-self.EOT_vibrance/2
        vibrance_val_temp=self.vibrance_val+change_val*eot_size

        filtered_hsv=rgb2hsv_torch(filtered)
        filtered_h=filtered_hsv[:,0:1,:,:]
        filtered_s=filtered_hsv[:,1:2,:,:]
        cur_vibrance_val=torch.clamp(vibrance_val_temp,0,2)
        filtered_s=1/(1+torch.exp(-1*cur_vibrance_val*10*(filtered_s-0.5)))
        filtered_v=filtered_hsv[:,2:3,:,:]
        filtered_hsv=torch.cat([filtered_h,filtered_s,filtered_v],1)
        filtered_hsv=torch.clamp(filtered_hsv,0,1)
        filtered=hsv2rgb_torch(filtered_hsv)
        filtered=torch.clamp(filtered,0,1)

        change_val=self.EOT_color_temp_r_b*np.random.rand()-self.EOT_color_temp_r_b/2
        color_temp_r_b_temp=self.color_temp_r_b+change_val*eot_size

        cur_ct=torch.clamp(color_temp_r_b_temp,-1,1)
        filtered_r=filtered[:,0:1,:,:]
        filtered_g=filtered[:,1:2,:,:]
        filtered_b=filtered[:,2:3,:,:]
        filtered_r=filtered_r+cur_ct[0]
        filtered_g=filtered_g+cur_ct[1]
        filtered_b=filtered_b+cur_ct[2]
        filtered=torch.cat([filtered_r,filtered_g,filtered_b],1)
        filtered=torch.clamp(filtered,0,1)



        return filtered

    def inspect(self,file):


        input_size=[233,160]
        input = Image.open(file)
        input = np.asarray(input)
        input=input/255
        input=np.transpose(input,[2,0,1])
        input = torch.from_numpy(input)
        input=torch.unsqueeze(input,0).float()
        resize_transform = T.Resize(size = input_size)
        input=resize_transform(input)
        result=self.run_filter_sharpen(input,resize_transform)

        result=torch.squeeze(result)
        result=torch.transpose(result,0,2)
        result=torch.transpose(result,0,1)
        print(torch.max(result),torch.min(result))
        result=torch.clamp(result,0,1)

        plt.imshow(result.detach().numpy())
        plt.show()



    def inspect_all(self,folder,distance,subfolder,eot_val=0,use_pixel=False):

        input_size=[233,160]

        input_base=folder+'2/'+subfolder+'/'
        input_files=os.listdir(input_base)
        label_base=folder+'/'+distance+'/'+subfolder+'/'
        label_files=os.listdir(label_base)

        resize_transform = T.Resize(size = input_size)

        avg_loss=0

        fig, axs = plt.subplots(2, len(input_files))

        for i in range(len(input_files)):
            file=input_base+input_files[i]
            input = Image.open(file)
            input = np.asarray(input)
            input=input/255
            input=np.transpose(input,[2,0,1])
            input = torch.from_numpy(input)
            input=torch.unsqueeze(input,0).float()
            input=resize_transform(input)
            result=self.run_filter_sharpen(input,resize_transform,eot_val,use_pixel)
            result=resize_transform(result)

            result=torch.squeeze(result)
            result=torch.transpose(result,0,2)
            result=torch.transpose(result,0,1)
            result=torch.clamp(result,0,1)

            label=label_base+label_files[i]
            label = Image.open(label)
            label = np.asarray(label)
            label=label/255
            label=np.transpose(label,[2,0,1])
            label = torch.from_numpy(label)
            label=torch.unsqueeze(label,0).float()
            label=resize_transform(label)

            label=torch.squeeze(label)
            label=torch.transpose(label,0,2)
            label=torch.transpose(label,0,1)
            label=torch.clamp(label,0,1)

            print(result.shape,label.shape)

            axs[0, i].imshow(result.detach().numpy())
            axs[1, i].imshow(label.detach().numpy())
        plt.show()


def form_train_test(source_dir,input,output,train_dirs,input_size):
    train_data=[]

    cur_input_base=source_dir+input
    cur_output_base=source_dir+output
    for train_index in range(len(train_dirs)):
        cur_input_folder=cur_input_base+'/'+train_dirs[train_index]
        cur_input_files=os.listdir(cur_input_folder)

        cur_output_folder=cur_output_base+'/'+train_dirs[train_index]

        for file_dir in range(len(cur_input_files)):
            cur_input_file_dir=cur_input_folder+'/'+cur_input_files[file_dir]
            out_file_name=cur_input_files[file_dir]
            out_file_name=out_file_name[0:-4]
            out_file_name=out_file_name+'.JPG'
            cur_output_file_dir=cur_output_folder+'/'+out_file_name


            temp_list=[cur_input_file_dir,cur_output_file_dir]
            temp_out=[]

            for temp_index in range(len(temp_list)):
                input = Image.open(temp_list[temp_index])
                input = np.asarray(input)
                input=input/255
                input=np.transpose(input,[2,0,1])
                input = torch.from_numpy(input)
                input=torch.unsqueeze(input,0).float()
                resize_transform = T.Resize(size = input_size)
                input=resize_transform(input)
                temp_out.append(input.numpy())
            train_data.append(temp_out)

    return train_data


def try_forward(mode,device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes,eot_val=0):


    blur_width=1.1509
    blur_temp=[10, 10, 10, 10, 10, 10, 10]
    blur_stride=[1,2,3,5,7,9,12]

    blur_width2=1.0781
    blur_temp2=10

    blur_height=1
    blur_channels=3


    sharp_type='cross'
    sharp_k_size=3

    N_exp=2.4895
    c_exp=5.5371
    lambda_divide=9.5015
    rbg_lambda=[63.0455, 57.2907, 45.5621]
    sky_rbg=[0.1,0.1,0.1]
    turbidity=2.1030
    sv_shift=[-0.0129, -0.3531]

    ds_k_size=3
    ds_channels=3
    ds_temperature=1
    ds_height=1
    ds_width=0.1
    ds_stride=1

    #contrast
    contrast_val=-0.0881

    #vibrance
    vibrance_val=0.9458

    #shadow highlight
    shahigh_val=[[0.1293, 0.9801]]

    #exposure
    exposure_val=0.5402

    #color temperature
    color_temp_r_b=[0.0559, 0.0735, 0.0901]


    lr=0.002
    epochs=50
    eval_epoch=5
    batch_size=20
    batch_size_test=batch_size
    weight_decay=0

    scale_x=[0.9,1]
    scale_y=[0.9,1]
    scale_interval=0.05

    all_intervals=[[[0,1],[0,1]]]

    sharpen_configs=[sharp_type, sharp_k_size]
    desharpen_configs=[[ds_k_size,ds_width,ds_temperature,ds_stride,ds_height, ds_channels],[1,2]]

    manual_keeps=[[[0,2],[0,2]],
                [[0,3],[0,3]],
                [[0,3],[0,1]],
                [[0,3],[0,1]],
                [[0,1],[0,1]],
                [[0,2],[0,1]],
                [[0,2],[0,1]]]

    k_a_ratio=[0.999,0.001]


    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./weights_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))


    blur_module_name='BlurModule_'+model_name
    blur_module=BlurModule(blur_stride,blur_width,blur_temp,blur_width2,blur_temp2,N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity, sv_shift, device,trainable=True,load=load, model_name=blur_module_name, num_days=len(data_dircts))


    style_module_name='StyleFilters_'+model_name
    filters=StyleFilters(sharpen_configs,desharpen_configs,device,contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name=style_module_name,load=load,manual_keep=manual_keeps,trainable=True)

    all_variables=[]
    for i in range(len(blur_module.variable_list)):
        all_variables.append(blur_module.variable_list[i])
    for i in range(len(filters.variable_list)):
        all_variables.append(filters.variable_list[i])

    optimizer = optim.Adam(all_variables, lr=lr, amsgrad=True,weight_decay=weight_decay)

    train_data_list=[]
    test_data_list=[]
    for d_index in range(len(data_dircts)):
        source_dir=data_dircts[d_index]
        input='digital'
        outputs=['2','6','12','18','24','32','40']
        output=[]
        for tempi in range(len(distance_indexes[d_index])):
            output.append(outputs[distance_indexes[d_index][tempi]])
        train_dirs=data_folders[d_index]
        test_dirs=['inspect']

        input_size=[233,160]
        out_ratio=1
        output_size=[int(input_size[0]/out_ratio),int(input_size[1]/out_ratio)]
        train_data_list_=[]
        test_data_list_=[]
        for data_i in range(len(output)):
            train_data=form_train_test(source_dir,input,output[data_i],train_dirs,input_size)
            test_data=form_train_test(source_dir,input,output[data_i],test_dirs,input_size)
            train_data=torch.tensor(train_data)
            test_data=torch.tensor(test_data)
            train_data_list_.append(train_data)
            test_data_list_.append(test_data)

        train_data_list_=torch.stack(train_data_list_)
        test_data_list_=torch.stack(test_data_list_)
        train_data_list.append(train_data_list_)
        test_data_list.append(test_data_list_)


    if mode=='manual':
        distance_index=5
        img_index=2
        day_index=0
        input_=train_data_list[day_index][distance_index][img_index][0]
        label=train_data_list[day_index][distance_index][img_index][1]
        input=run_filter_full(color_network,blur_module,filters,distance_index,input_.to(device),day_index=day_index,eot_val=eot_val,sky_color=sky_rbg_list[day_index])


        #-------------plot results---------
        fig, axs = plt.subplots(1, 3)

        input_=torch.squeeze(input_)
        input_=torch.transpose(input_,0,2)
        input_=torch.transpose(input_,0,1)
        input_=torch.clamp(input_,0,1)

        input=torch.squeeze(input)
        input=torch.transpose(input,0,2)
        input=torch.transpose(input,0,1)
        input=torch.clamp(input,0,1)

        label=torch.squeeze(label)
        label=torch.transpose(label,0,2)
        label=torch.transpose(label,0,1)
        label=torch.clamp(label,0,1)

        if device=='cuda':
            input_=input_.cpu()
            input=input.cpu()
            label=label.cpu()

        axs[0].imshow(input_.detach().numpy())
        axs[1].imshow(input.detach().numpy())
        axs[2].imshow(label.detach().numpy())
        plt.show()

        return
    elif mode=='inspect':
        base_dirct='./inspect_output/'+model_name
        if os.path.exists(base_dirct)==False:
            os.makedirs(base_dirct)
        for ins_index in range(len(inspect_indexes)):
            folder_name=data_dircts[inspect_indexes[ins_index]]
            folder_name=folder_name.split('/')
            folder_name=folder_name[-2]
            cur_dir=base_dirct+'/'+folder_name
            print('folder name:',folder_name)
            if os.path.exists(cur_dir)==False:
                os.makedirs(cur_dir)
            for distance_index in range(test_data_list[inspect_indexes[ins_index]].shape[0]):
                print('distance_index:',distance_index)
                fig, axs = plt.subplots(3, test_data_list[inspect_indexes[ins_index]].shape[1])
                for img_index in range(test_data_list[inspect_indexes[ins_index]].shape[1]):
                    input_=test_data_list[inspect_indexes[ins_index]][distance_index][img_index][0]
                    label=test_data_list[inspect_indexes[ins_index]][distance_index][img_index][1]

                    input=run_filter_full(color_network,blur_module,filters,distance_indexes[inspect_indexes[ins_index]][distance_index],input_.to(device),day_index=inspect_indexes[ins_index],eot_val=eot_val,sky_color=sky_rbg_list[inspect_indexes[ins_index]])


                    #-------------plot results---------

                    input_=torch.squeeze(input_)
                    input_=torch.transpose(input_,0,2)
                    input_=torch.transpose(input_,0,1)
                    input_=torch.clamp(input_,0,1)

                    input=torch.squeeze(input)
                    input=torch.transpose(input,0,2)
                    input=torch.transpose(input,0,1)
                    input=torch.clamp(input,0,1)

                    label=torch.squeeze(label)
                    label=torch.transpose(label,0,2)
                    label=torch.transpose(label,0,1)
                    label=torch.clamp(label,0,1)

                    if device=='cuda':
                        input_=input_.cpu()
                        input=input.cpu()
                        label=label.cpu()

                    axs[0,img_index].imshow(input_.detach().numpy())
                    axs[1,img_index].imshow(input.detach().numpy())
                    axs[2,img_index].imshow(label.detach().numpy())

                img_save_path=cur_dir+'/'+'distance'+str(distance_index)+'.jpg'

                my_dpi=200
                plt.savefig(img_save_path, dpi=my_dpi * 10)



def run_filter_full(color_network,blur_module,filters,distance_index,input,day_index,eot_val=0,sky_color='default'):
    #------------preparation---------------
    resize_final = T.Resize(size = [input.shape[2],input.shape[3]])

    #------------run color mapping network-----------
    input=torch.squeeze(input)
    apts=input.shape
    input=torch.transpose(input,0,1)
    input=torch.transpose(input,1,2)
    input=torch.reshape(input,[apts[1]*apts[2],apts[0]])

    input=color_network(input)

    input=torch.reshape(input,[apts[1],apts[2],apts[0]])
    input=torch.transpose(input,1,2)
    input=torch.transpose(input,0,1)
    input=torch.unsqueeze(input,0)

    #------------run blur------------
    if sky_color=='default':
        input=blur_module.run_blurring(input,distance_index,eot_size=eot_val,day_index=day_index)
    else:
        input=blur_module.run_blurring(input,distance_index,eot_size=eot_val,day_index=day_index,sky_rbg=sky_color)

    #------------run manual keep-------------
    input = filters.run_manual_crop(input,distance_index)

    #------------run style filters-------------
    input = filters.run_style_filter(input,eot_val)

    #------------run sharpen-------------
    input = filters.run_sharpen(input,eot_val)

    #------------resize back-------------
    input = resize_final(input)

    return input

def train_test_filter_ensemble(device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes):

    blur_width=0.8
    blur_temp=[10, 10, 10, 10, 10, 10, 10]
    blur_stride=[1,2,3,5,7,9,12]

    blur_width2=0.8
    blur_temp2=10

    blur_height=1
    blur_channels=3

    sharp_type='cross'
    sharp_k_size=3

    N_exp=2
    c_exp=5.1
    lambda_divide=10
    rbg_lambda=[63,57,45]
    sky_rbg=[0.1,0.1,0.1]
    turbidity=2.5
    sv_shift=[ -0.07, -0.17]

    ds_k_size=3
    ds_channels=3
    ds_temperature=1
    ds_height=1
    ds_width=0.1
    ds_stride=1

    #contrast
    contrast_val=0.2

    #vibrance
    vibrance_val=0.2

    #shadow highlight
    shahigh_val=[[0.2,0.8]]

    #exposure
    exposure_val=0.2

    #color temperature
    color_temp_r_b=[0,0,0]


    lr=0.002
    epochs=50
    eval_epoch=5
    batch_size=20
    batch_size_test=batch_size
    weight_decay=0

    scale_x=[0.9,1]
    scale_y=[0.9,1]
    scale_interval=0.05

    all_intervals=[[[0,1],[0,1]]]

    sharpen_configs=[sharp_type, sharp_k_size]
    desharpen_configs=[[ds_k_size,ds_width,ds_temperature,ds_stride,ds_height, ds_channels],[1,2]]

    manual_keeps=[[[0,2],[0,2]],
                [[0,3],[0,3]],
                [[0,3],[0,1]],
                [[0,3],[0,1]],
                [[0,1],[0,1]],
                [[0,2],[0,1]],
                [[0,2],[0,1]]]

    k_a_ratio=[0.999,0.001]

    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./weights_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))


    blur_module_name='BlurModule_'+model_name
    blur_module=BlurModule(blur_stride,blur_width,blur_temp,blur_width2,blur_temp2, N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity, sv_shift, device,trainable=True,load=load, model_name=blur_module_name,num_days=len(data_dircts))


    style_module_name='StyleFilters_'+model_name
    filters=StyleFilters(sharpen_configs,desharpen_configs,device,contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name=style_module_name,load=load,manual_keep=manual_keeps,trainable=True)

    all_variables=[]
    for i in range(len(blur_module.variable_list)):
        all_variables.append(blur_module.variable_list[i])
    for i in range(len(filters.variable_list)):
        all_variables.append(filters.variable_list[i])

    optimizer = optim.Adam(all_variables, lr=lr, amsgrad=True,weight_decay=weight_decay)

    train_data_list=[]
    test_data_list=[]
    for d_index in range(len(data_dircts)):
        source_dir=data_dircts[d_index]
        input='digital'
        outputs=['2','6','12','18','24','32','40']
        output=[]
        for tempi in range(len(distance_indexes[d_index])):
            output.append(outputs[distance_indexes[d_index][tempi]])
        train_dirs=data_folders[d_index]
        test_dirs=['inspect']

        input_size=[233,160]
        out_ratio=1
        output_size=[int(input_size[0]/out_ratio),int(input_size[1]/out_ratio)]
        train_data_list_=[]
        test_data_list_=[]
        for data_i in range(len(output)):
            train_data=form_train_test(source_dir,input,output[data_i],train_dirs,input_size)
            test_data=form_train_test(source_dir,input,output[data_i],test_dirs,input_size)
            train_data=torch.tensor(train_data).to(device)
            test_data=torch.tensor(test_data).to(device)
            train_data_list_.append(train_data)
            test_data_list_.append(test_data)

        train_data_list_=torch.stack(train_data_list_)
        test_data_list_=torch.stack(test_data_list_)
        train_data_list.append(train_data_list_)
        test_data_list.append(test_data_list_)

    for epoch in range(epochs):

        total=0
        for tempi in range(len(train_data_list)):
            cur_total=train_data_list[tempi].shape[0]*train_data_list[tempi].shape[1]
            total=total+cur_total
        num_batches=int(total/batch_size)

        avg_train_loss=0

        for batch_index in range(num_batches):


            batch_input_list=[]
            batch_label_list=[]
            distance_index_list=[]
            day_index_list=[]
            for day_index in range(len(train_data_list)):
                dis_index=int(np.random.rand()*train_data_list[day_index].shape[0])
                dis_counter=distance_indexes[day_index][dis_index]
                s_index=int(np.random.rand()*train_data_list[day_index][dis_index].shape[0])
                cur_input=train_data_list[day_index][dis_index,s_index,0,:,:,:,:]
                cur_label=train_data_list[day_index][dis_index,s_index,1,:,:,:,:]
                batch_input_list.append(cur_input)
                batch_label_list.append(cur_label)
                distance_index_list.append(dis_counter)
                day_index_list.append(day_index)
            max_distance_num=0
            for i in range(len(train_data_list)):
                if train_data_list[i].shape[0]>max_distance_num:
                    max_distance_num=train_data_list[i].shape[0]
            for day_index in range(len(train_data_list)):
                if train_data_list[day_index].shape[0]==max_distance_num:
                    for dis_index in range(train_data_list[day_index].shape[0]):
                        dis_counter=distance_indexes[day_index][dis_index]
                        s_index=int(np.random.rand()*train_data_list[day_index][dis_index].shape[0])
                        cur_input=train_data_list[day_index][dis_index,s_index,0,:,:,:,:]
                        cur_label=train_data_list[day_index][dis_index,s_index,1,:,:,:,:]
                        batch_input_list.append(cur_input)
                        batch_label_list.append(cur_label)
                        distance_index_list.append(dis_counter)
                        day_index_list.append(day_index)
                    break
            remain=batch_size-len(distance_index_list)
            for temp_i in range(remain):
                day_index=int(np.random.rand()*len(train_data_list))
                dis_index=int(np.random.rand()*train_data_list[day_index].shape[0])
                dis_counter=distance_indexes[day_index][dis_index]
                s_index=int(np.random.rand()*train_data_list[day_index][dis_index].shape[0])
                cur_input=train_data_list[day_index][dis_index,s_index,0,:,:,:,:]
                cur_label=train_data_list[day_index][dis_index,s_index,1,:,:,:,:]
                batch_input_list.append(cur_input)
                batch_label_list.append(cur_label)
                distance_index_list.append(dis_counter)
                day_index_list.append(day_index)
            batch_input=torch.cat(batch_input_list,0)
            batch_label=torch.cat(batch_label_list,0)
            batch_input=torch.unsqueeze(batch_input,1)
            batch_label=torch.unsqueeze(batch_label,1)

            resize_transform = T.Resize(size = input_size)
            resize_transform_out = T.Resize(size = output_size)

            results=[]
            for i in range(batch_input.shape[0]):

                result=run_filter_full(color_network,blur_module,filters,distance_index_list[i],batch_input[i].to(device),day_index=day_index_list[i],sky_color=sky_rbg_list[day_index_list[i]])
                results.append(result)
            results=torch.stack(results,0)

            results_cropped=[]
            batch_label_cropped=[]
            for crop_index in range(len(all_intervals)):
                rs=results.shape
                cur_result_cropped=results[:,:,:,int(rs[3]*all_intervals[crop_index][0][0]):int(rs[3]*all_intervals[crop_index][0][1]),int(rs[4]*all_intervals[crop_index][1][0]):int(rs[4]*all_intervals[crop_index][1][1])]
                bls=batch_label.shape
                cur_bl_cropped=batch_label[:,:,:,int(bls[3]*all_intervals[crop_index][0][0]):int(bls[3]*all_intervals[crop_index][0][1]),int(bls[4]*all_intervals[crop_index][1][0]):int(bls[4]*all_intervals[crop_index][1][1])]

                cur_result_cropped=torch.squeeze(cur_result_cropped)
                cur_result_cropped=resize_transform_out(cur_result_cropped)
                cur_result_cropped=torch.unsqueeze(cur_result_cropped,1)
                cur_bl_cropped=torch.squeeze(cur_bl_cropped)
                cur_bl_cropped=resize_transform_out(cur_bl_cropped)
                cur_bl_cropped=torch.unsqueeze(cur_bl_cropped,1)

                results_cropped.append(cur_result_cropped)
                batch_label_cropped.append(cur_bl_cropped)

            results_cropped=torch.cat(results_cropped,1)
            batch_label_cropped=torch.cat(batch_label_cropped,1)

            rcs=results_cropped.shape
            results_cropped=torch.reshape(results_cropped,[rcs[0],rcs[1],rcs[2]*rcs[3]*rcs[4]])

            blcs=batch_label_cropped.shape
            batch_label_cropped=torch.reshape(batch_label_cropped,[blcs[0],blcs[1],blcs[2]*blcs[3]*blcs[4]])
            batch_label_cropped=torch.transpose(batch_label_cropped,1,2)

            results_cropped=torch.unsqueeze(results_cropped,3)
            batch_label_cropped=torch.unsqueeze(batch_label_cropped,1)
            loss=results_cropped-batch_label_cropped
            loss=loss*loss
            loss=torch.mean(loss,2)
            loss=torch.reshape(loss,[loss.shape[0],loss.shape[1]*loss.shape[2]])
            loss,_=torch.min(loss,1)
            loss=torch.mean(loss)

            avg_train_loss=avg_train_loss+loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss=avg_train_loss/num_batches

        if epoch%eval_epoch==0:
            filters.save(epoch)
            blur_module.save(epoch)

            if device=='cuda':
                avg_train_loss=avg_train_loss.cpu()
            print('train -- epoch: '+str(epoch)+' loss: '+str(avg_train_loss.detach().numpy())+'\n')

def print_all_parameters(device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes):
    blur_width=0.8
    blur_temp=[10, 10, 10, 10, 10, 10, 10]
    blur_stride=[1,2,3,5,7,9,12]

    blur_width2=0.8
    blur_temp2=10

    blur_height=1
    blur_channels=3

    #sharpen
    sharp_type='cross'
    sharp_k_size=3

    N_exp=2
    c_exp=5.1
    lambda_divide=10
    rbg_lambda=[63,57,45]
    sky_rbg=[0.1,0.1,0.1]
    turbidity=2.5
    sv_shift=[ -0.07, -0.17]

    #desharpen
    ds_k_size=3
    ds_channels=3
    ds_temperature=1
    ds_height=1
    ds_width=0.1
    ds_stride=1

    #contrast
    contrast_val=0.2

    #vibrance
    vibrance_val=0.2

    #shadow highlight
    shahigh_val=[[0.2,0.8]]

    #exposure
    exposure_val=0.2

    #color temperature
    color_temp_r_b=[0,0,0]


    lr=0.002
    epochs=50
    eval_epoch=5
    batch_size=20
    batch_size_test=batch_size
    weight_decay=0

    scale_x=[0.9,1]
    scale_y=[0.9,1]
    scale_interval=0.05

    all_intervals=[[[0,1],[0,1]]]

    sharpen_configs=[sharp_type, sharp_k_size]
    desharpen_configs=[[ds_k_size,ds_width,ds_temperature,ds_stride,ds_height, ds_channels],[1,2]]

    manual_keeps=[[[0,2],[0,2]],
                [[0,3],[0,3]],
                [[0,3],[0,1]],
                [[0,3],[0,1]],
                [[0,1],[0,1]],
                [[0,2],[0,1]],
                [[0,2],[0,1]]]

    k_a_ratio=[0.999,0.001]

    network_width=300
    color_network = MappingNet(network_width).to(device)
    color_network.load_state_dict(torch.load('./weights_to_load/2023_3_1_color_mapping_network/version_2023_2_23_temp.pth'))


    blur_module_name='BlurModule_'+model_name
    blur_module=BlurModule(blur_stride,blur_width,blur_temp,blur_width2,blur_temp2,N_exp, c_exp, lambda_divide, rbg_lambda, sky_rbg, turbidity, sv_shift, device,trainable=True,load=load, model_name=blur_module_name, num_days=len(data_dircts))


    style_module_name='StyleFilters_'+model_name
    filters=StyleFilters(sharpen_configs,desharpen_configs,device,contrast_val,vibrance_val,shahigh_val,exposure_val,color_temp_r_b,model_name=style_module_name,load=load,manual_keep=manual_keeps,trainable=True)


    print('-------------BlurModule-------------')
    print('kernel_sizes:',blur_module.kernel_sizes)
    print('strides:',blur_module.strides)
    print('widths:',blur_module.widths)
    print('temperatures:',blur_module.temperatures)
    print('widths:',blur_module.widths2)
    print('temperatures:',blur_module.temperatures2)
    print('N_exp:',blur_module.N_exp)
    print('c_exp:',blur_module.c_exp)
    print('lambda_divide:',blur_module.lambda_divide)
    print('rbg_lambda:',blur_module.rbg_lambda)
    print('sky_rbg:',blur_module.sky_rbg)
    print('turbidity:',blur_module.turbidity)
    print('sv_shift:',blur_module.sv_shift)

    print('-------------StyleFilters-------------')
    print('sharpen_configs:',filters.sharpen_configs)
    print('desharpen_configs:',filters.desharpen_configs)
    print('contrast_val:',filters.contrast_val)
    print('vibrance_val:',filters.vibrance_val)
    print('shahigh_val:',filters.shahigh_val)
    print('exposure_val:',filters.exposure_val)
    print('color_temp_r_b:',filters.color_temp_r_b)

def loading_sky_rgb(data_dircts,sky_color_loc,device):
    result_list=[]
    for d_index in range(len(data_dircts)):
        cur_dirct=data_dircts[d_index]+sky_color_loc

        input = Image.open(cur_dirct)
        input = np.asarray(input)
        input=input/255

        inputr=np.mean(input[:,:,0])
        inputg=np.mean(input[:,:,1])
        inputb=np.mean(input[:,:,2])

        res=torch.tensor([inputr,inputg,inputb]).float().to(device)

        result_list.append(res)
    return result_list


if __name__=='__main__':
    model_name='demo'

    #those are high quality data manually selected
    data_dircts=['./process_photos/2023_5_16_cropped/',
                './process_photos/2023_5_26_cropped_undersun/',
                './process_photos/2023-2-28_atmospheric_perspective/',
                './process_photos/2023-3-7_atmospheric_perspective/',
                './process_photos/2023-5-29_atmospheric perspective data/']
    distance_indexes=[[0,1,2,3,4,5,6],
                        [0,3,6],
                        [0,1,2,3,4,5,6],
                        [0,1,2,3,4,5,6],
                        [1,3,5]]
    data_folders=[['color_train','inspect','plates'],
                  ['color_train','inspect'],
                  ['color_test','color_train'],
                  ['color_test','color_train'],
                  ['color_test','color_train','inspect']]
    inspect_indexes=[0,1,4]

    device='cuda'
    sky_color_loc='sky_color/sky_color.JPG'
    sky_rbg_list=loading_sky_rgb(data_dircts,sky_color_loc,device)

    #-----------inspect DIC with our configuration-----------
    '''DIC parameter optimized with the full training set has already been set as default'''
    load=False
    try_forward('inspect',device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes)

    #-----------sample code for train and inspect-------------
    '''
    Codes below are for optimizing and printing the resulting parameters.
    '''

    # load=False
    # train_test_filter_ensemble(device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes)
    # load=True
    # print_all_parameters(device,load,model_name,data_dircts,distance_indexes,data_folders,sky_rbg_list,inspect_indexes)



























