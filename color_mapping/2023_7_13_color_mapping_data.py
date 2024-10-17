import PIL.Image as Image
import numpy as np
import os
import torch
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt

def form_train_val(input_dir_name,input_files,input_suffix,label_rotates,label_dir_name,label_suffix,save_dir,sample_num=50,train_shape=[14,10,20],val_shape=[8,5,20]):
    result_train=[]#color_num,2,3
    result_val=[]#color_num,2,3

    for type in range(len(input_files)):
        for index in range(len(input_files[type])):
            if type==0:
                cur_shape=[int(train_shape[0]*train_shape[2]),int(train_shape[1]*train_shape[2])]
            else:
                cur_shape=[int(val_shape[0]*val_shape[2]),int(val_shape[1]*val_shape[2])]
            resize_transform = T.Resize(size = (cur_shape[0],cur_shape[1]))

            input = Image.open(input_dir_name+'/'+input_files[type][index]+input_suffix[type][index])
            input = np.asarray(input)
            label = Image.open(label_dir_name+'/'+input_files[type][index]+label_suffix)
            label = np.asarray(label)

            input=np.transpose(input,[2,0,1])
            input=input/255
            input=torch.tensor(input)
            input=resize_transform(input)#torch.Size([3, 280, 200])
            label=np.transpose(label,[2,0,1])
            label=label/255
            label=torch.tensor(label)
            label=torch.rot90(label,4-label_rotates[type][index],[1,2])
            label=resize_transform(label)#torch.Size([3, 280, 200])

            # if type==1 and index==3:
            #     input=input.numpy()
            #     input=np.transpose(input,[1,2,0])
            #     #input=input*255
            #     label=label.numpy()
            #     label=np.transpose(label,[1,2,0])
            #     #label=label*255
            #
            #     to_plot=[input,label]
            #     f = plt.figure()
            #     n=len(to_plot)
            #     for i in range(n):
            #         # Debug, plot figure
            #         f.add_subplot(1, n, i + 1)
            #         plt.imshow(to_plot[i])
            #     plt.show()
            #     return

            sample_list=[]#[all positions,sample index, xy]
            if type==0:
                y_num=train_shape[0]
                x_num=train_shape[1]
                interval=train_shape[2]
            else:
                y_num=val_shape[0]
                x_num=val_shape[1]
                interval=val_shape[2]
            for y_index in range(y_num):
                for x_index in range(x_num):
                    cur_list=[]
                    y_start=y_index*interval
                    y_end=(y_index+1)*interval
                    x_start=x_index*interval
                    x_end=(x_index+1)*interval
                    for s_index in range(sample_num):
                        cur_yx=[int(y_start+np.random.rand()*(y_end-y_start)),int(x_start+np.random.rand()*(x_end-x_start))]
                        cur_list.append(cur_yx)
                    sample_list.append(cur_list)

            temp_res=[]
            for temp_res_i in range(len(sample_list)):
                avg_in_color=[]
                avg_out_color=[]
                for temp_res_j in range(len(sample_list[temp_res_i])):
                    avg_in_color.append(input[:,sample_list[temp_res_i][temp_res_j][0],sample_list[temp_res_i][temp_res_j][1]].numpy())
                    avg_out_color.append(label[:,sample_list[temp_res_i][temp_res_j][0],sample_list[temp_res_i][temp_res_j][1]].numpy())
                avg_in_color=np.asarray(avg_in_color)
                avg_out_color=np.asarray(avg_out_color)
                avg_in_color=np.average(avg_in_color,0)
                avg_out_color=np.average(avg_out_color,0)
                if type==0:
                    result_train.append([avg_in_color,avg_out_color])
                if type==1:
                    result_val.append([avg_in_color,avg_out_color])

    result_train=np.asarray(result_train)
    result_val=np.asarray(result_val)

    print(result_train.shape)
    print(result_val.shape)

    # inspect_in=[]
    # inspect_out=[]
    # inspect_type=0
    # ins_y=10
    # ins_x=5
    # ins_list=[result_train,result_val]
    # ins_in=[]
    # ins_out=[]
    # for y in range(ins_y):
    #     list_in=[]
    #     list_out=[]
    #     for x in range(ins_x):
    #         index=int(ins_list[inspect_type].shape[0]*np.random.rand())
    #         cur_in=ins_list[inspect_type][index][0]
    #         cur_lab=ins_list[inspect_type][index][1]
    #         list_in.append(cur_in)
    #         list_out.append(cur_lab)
    #     ins_in.append(list_in)
    #     ins_out.append(list_out)
    # ins_in=np.asarray(ins_in)
    # ins_out=np.asarray(ins_out)
    # ins_in=torch.tensor(ins_in)
    # ins_out=torch.tensor(ins_out)
    # ins_in=torch.repeat_interleave(ins_in,5,dim=0)
    # ins_in=torch.repeat_interleave(ins_in,5,dim=1)
    # ins_out=torch.repeat_interleave(ins_out,5,dim=0)
    # ins_out=torch.repeat_interleave(ins_out,5,dim=1)
    # ins_in=ins_in.numpy()
    # ins_out=ins_out.numpy()
    # to_plot=[ins_in,ins_out]
    # f = plt.figure()
    # n=len(to_plot)
    # for i in range(n):
    #     # Debug, plot figure
    #     f.add_subplot(1, n, i + 1)
    #     plt.imshow(to_plot[i])
    # plt.show()
    # return


    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)
    np.save(save_dir+'/'+'train.npy',result_train)
    np.save(save_dir+'/'+'val.npy',result_val)


if __name__=='__main__':

    input_dir_name='./color mapping/1 - 2023-7-6 - selected'
    input_files=[['p1','p6','p11','p12','p13','p14','p15','p16'],['p7','p8','p9','p10']]
    input_suffix=[['.jpeg','.jpeg','.jpeg','.jpeg','.jpeg','.jpeg','.jpeg','.jpeg'],['.JPG','.JPG','.JPG','.JPG']]

    label_rotates=[[1,1,3,1,1,2,2,1,],[2,0,1,0]]
    label_dir_name='./color mapping/1 - 2023-7-6 - photos'
    label_suffix='.JPG'


    save_dir='./color mapping/1 - 2023-7-6 - np'

    form_train_val(input_dir_name,input_files,input_suffix,label_rotates,label_dir_name,label_suffix,save_dir)