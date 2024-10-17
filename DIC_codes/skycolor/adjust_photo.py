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
    #对出界值的处理
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

def try_parameters(file):
    '''
    input = Image.open(temp_list[temp_index])
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()
    resize_transform = T.Resize(size = input_size)
    input=resize_transform(input)
    '''

    input = Image.open(file)
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()

    # input=input[:,3:4,:,:]
    #
    # input=torch.squeeze(input,0)
    # input=torch.transpose(input,0,2)
    # input=torch.transpose(input,0,1)
    #
    # plt.imshow(input.detach().numpy())
    # plt.show()

    sx=1.4
    vx=1.2

    fig, axs = plt.subplots(1, 2)

    #input[:,3,:,:] is the transparent channel, 0 for transparent part, 1 for part with color. The first three channels are the RGB channels.

    input_=input[:,0:3,:,:]
    input_hsv=rgb2hsv_torch(input_.clone())
    input_h=input_hsv[:,0:1,:,:]
    input_s=input_hsv[:,1:2,:,:]*sx
    input_v=input_hsv[:,2:3,:,:]*vx
    input_hsv=torch.cat([input_h,input_s,input_v],1)
    input_hsv=torch.clamp(input_hsv,0,1)
    input=hsv2rgb_torch(input_hsv)

    input_=torch.squeeze(input_,0)
    input_=torch.transpose(input_,0,2)
    input_=torch.transpose(input_,0,1)

    input=torch.squeeze(input,0)
    input=torch.transpose(input,0,2)
    input=torch.transpose(input,0,1)

    axs[0].imshow(input_.detach().numpy())
    axs[1].imshow(input.detach().numpy())
    plt.show()

def calculate_sky_avg(file):
    input = Image.open(file)
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()

    sx=1.4
    vx=1.5

    input_ori=input[:,0:3,:,:].clone()
    input_ind=input[:,3:4,:,:].clone()

    input_hsv=rgb2hsv_torch(input_ori.clone())
    input_h=input_hsv[:,0:1,:,:]
    input_s=input_hsv[:,1:2,:,:]*sx
    input_v=input_hsv[:,2:3,:,:]*vx
    input_hsv=torch.cat([input_h,input_s,input_v],1)
    input_hsv=torch.clamp(input_hsv,0,1)
    input=hsv2rgb_torch(input_hsv)

    pixel_amount=torch.sum(input_ind)
    input_ind_=torch.Tensor.repeat(input_ind,[1,3,1,1])
    three_channel_sum=input*input_ind_
    three_channel_sum=torch.sum(three_channel_sum,2)
    three_channel_sum=torch.sum(three_channel_sum,2)
    three_channel_sum=three_channel_sum/pixel_amount
    three_channel_sum=torch.squeeze(three_channel_sum)#tensor([0.5732, 0.7222, 1.0000])
    return three_channel_sum

def precalc_sky_avgs(folder):
    files=os.listdir(folder)
    file_names=[]
    for i in range(len(files)):
        cur_name=files[i][:-4]
        file_names.append(cur_name)
    result_dict={}
    for i in range(len(file_names)):
        cur_parts=file_names[i].split('_')
        vis=cur_parts[0]
        elev=cur_parts[1]
        alb=cur_parts[2]
        if (vis in result_dict.keys())==False:
            result_dict[vis]=[]
        three_channel_sum=calculate_sky_avg(folder+'/'+file_names[i]+'.PNG')
        three_channel_sum=three_channel_sum.numpy()
        to_add=[elev,alb,three_channel_sum]
        result_dict[vis].append(to_add)
    cur_keys=list(result_dict.keys())

    cloudy_skies=[[0.7006, 0.7253, 0.7282],
                [0.8753, 0.8753, 0.8753],
                [0.6753, 0.6753, 0.6753]]
    for i in range(len(cur_keys)):
        vis=cur_keys[i]
        elev='cloudy'
        alb='cloudy'
        for j in range(len(cloudy_skies)):
            three_channel_sum=cloudy_skies[j]
            to_add=[elev,alb,three_channel_sum]
            result_dict[vis].append(to_add)
    return result_dict

def calculate_sky_white(file,pixels):
    input = Image.open(file)
    input = np.asarray(input)
    input=input/255
    input=np.transpose(input,[2,0,1])
    input = torch.from_numpy(input)
    input=torch.unsqueeze(input,0).float()

    sx=1.4
    vx=1.2

    input_ori=input[:,0:3,:,:].clone()
    input_ind=input[:,3:4,:,:].clone()

    input_hsv=rgb2hsv_torch(input_ori.clone())
    input_h=input_hsv[:,0:1,:,:]
    input_s=input_hsv[:,1:2,:,:]*sx
    input_v=input_hsv[:,2:3,:,:]*vx
    input_hsv=torch.cat([input_h,input_s,input_v],1)
    input_hsv=torch.clamp(input_hsv,0,1)
    input=hsv2rgb_torch(input_hsv)#torch.Size([1, 3, 107, 107])

    mid=int(input.shape[3]/2)
    start=input.shape[2]-pixels
    end=input.shape[2]
    result=input_ori[:,:,start:end,mid]
    result=torch.squeeze(result)
    result=torch.mean(result,1)
    return result

def precalc_sky_whites(folder,pixels):
    files=os.listdir(folder)
    file_names=[]
    for i in range(len(files)):
        cur_name=files[i][:-4]
        file_names.append(cur_name)
    result_dict={}
    for i in range(len(file_names)):
        cur_parts=file_names[i].split('_')
        vis=cur_parts[0]
        elev=cur_parts[1]
        alb=cur_parts[2]
        if (vis in result_dict.keys())==False:
            result_dict[vis]=[]
        three_channel_sum=calculate_sky_white(folder+'/'+file_names[i]+'.PNG',pixels)
        three_channel_sum=three_channel_sum.numpy()
        to_add=[elev,alb,three_channel_sum]
        result_dict[vis].append(to_add)
    cur_keys=list(result_dict.keys())
    return result_dict

if __name__=='__main__':
    # file='./images/131_90_5.PNG'
    # try_parameters(file)

    # file='./images/131_90_10.PNG'
    # calculate_sky_avg(file)

    # folder='./images'
    # result_dict=precalc_sky_avgs(folder)
    # print(result_dict['71'])

    # file='./images/131_90_5.PNG'
    # pixels=4
    # result=calculate_sky_white(file,pixels)
    # print(result)

    folder='./images_albedo5'
    pixels=4
    result_dict=precalc_sky_whites(folder,pixels)
    print(result_dict)


























