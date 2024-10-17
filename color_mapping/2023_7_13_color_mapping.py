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

class PolyMapper:
    def __init__(self, degree=3,name='default',device='cpu',load=False,trainable=True):

        self.name=name
        self.device=device
        self.trainable=trainable

        load_index=0
        self.variable_list=[]
        self.load=load

        self.degree=degree
        self.degrees=self.count_of_ways(self.degree)
        self.degrees=torch.tensor(np.asarray(self.degrees)).to(device)

        self.weight=torch.rand([self.degrees.shape[0],1]).to(device)
        self.weight,load_index,self.variable_list=self.make_variable_differentiable(self.weight,self.load,load_index,self.variable_list)

        self.weight_matrix=torch.rand([3,3]).to(device)
        self.weight_matrix,load_index,self.variable_list=self.make_variable_differentiable(self.weight_matrix,self.load,load_index,self.variable_list)

        self.bias=torch.rand([1,3]).to(device)
        self.bias,load_index,self.variable_list=self.make_variable_differentiable(self.bias,self.load,load_index,self.variable_list)

    def forward(self,input):
        #input should be in the shape of (1,3,x,y)
        ori_shape=input.shape
        input=torch.reshape(input,[ori_shape[1],ori_shape[2]*ori_shape[3]])
        input=torch.unsqueeze(input,0)
        input=torch.Tensor.repeat(input,[self.degrees.shape[0],1,1])

        cur_degrees=torch.unsqueeze(self.degrees,2)

        input=input**cur_degrees

        cur_weight=torch.unsqueeze(self.weight,2)
        cur_weight=torch.Tensor.repeat(cur_weight,[1,input.shape[1],1])

        input=input*cur_weight

        input=torch.sum(input,0)

        input=torch.transpose(input,0,1)

        res=torch.matmul(input,self.weight_matrix)

        res=res+self.bias
        res=torch.transpose(res,0,1)

        res=torch.reshape(res,ori_shape)

        return res




    def save(self,epoch):
        save_dir='./model/'+self.name
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

    def make_variable_differentiable(self,variable,load,load_index,variable_list):
        if load==True:
            value=np.load('./model/'+self.name+'/'+str(load_index)+'.npy')
            variable=value
            load_index+=1
        variable = torch.tensor(variable).float()
        variable = variable.to(self.device)
        if self.trainable==True:
            variable.requires_grad_(True)
        variable_list.append(variable)
        return variable,load_index,variable_list


    def count_of_ways(self,n):
        results=[]

        count = 0
        for i in range(0, n+1):
            for j in range(0, n+1):
                for k in range(0, n+1):
                    if(i + j + k == n):
                        count = count + 1
                        results.append([i,j,k])
        return results

    def inspect_patch(self,input_name):
        input = Image.open(input_name)
        input = np.asarray(input)
        input=np.transpose(input,[2,0,1])
        input=torch.tensor(input)
        input=torch.unsqueeze(input,0)
        input=input/255
        res=self.forward(input)

        input=torch.squeeze(input).numpy()
        input=np.transpose(input,[1,2,0])
        #input=input*255
        res=torch.squeeze(res).detach().numpy()
        res=np.transpose(res,[1,2,0])
        #label=label*255

        to_plot=[input,res]
        f = plt.figure()
        n=len(to_plot)
        for i in range(n):
            # Debug, plot figure
            f.add_subplot(1, n, i + 1)
            plt.imshow(to_plot[i])
        plt.show()

def get_batch(data,bs):
    total=bs[0]*bs[1]
    res_in=[]
    res_lab=[]
    for i in range(total):
        cur_index=int(np.random.rand()*data.shape[0])
        cur_input=data[cur_index][0]
        cur_label=data[cur_index][1]
        res_in.append(cur_input)
        res_lab.append(cur_label)
    res_in=np.asarray(res_in)
    res_lab=np.asarray(res_lab)

    res_in=np.reshape(res_in,[1,res_in.shape[1],bs[0],bs[1]])
    res_lab=np.reshape(res_lab,[1,res_lab.shape[1],bs[0],bs[1]])

    return res_in,res_lab

def train_test(mapper):

    lr=0.002
    epochs=2000
    eval_epoch=300
    batch_size=[10,5]
    batch_size_test=[10,5]
    weight_decay=0

    train_data_dir='./color mapping/1 - 2023-7-6 - np/train.npy'
    val_data_dir='./color mapping/1 - 2023-7-6 - np/val.npy'

    #mapper=PolyMapper(degree,name=name,device=device,load=load)

    train_data=np.load(train_data_dir)
    val_data=np.load(val_data_dir)

    all_variables=[]
    for i in range(len(mapper.variable_list)):
        all_variables.append(mapper.variable_list[i])

    optimizer = optim.Adam(all_variables, lr=lr, amsgrad=True,weight_decay=weight_decay)

    for epoch in range(epochs):
        avg_train_loss=0
        num_batches=int(train_data.shape[0]/(batch_size[0]*batch_size[1]))
        for batch_index in range(num_batches):
            cur_in,cur_lab=get_batch(train_data,batch_size)
            cur_in=torch.tensor(cur_in).to(device).float()
            cur_lab=torch.tensor(cur_lab).to(device).float()
            cur_res=mapper.forward(cur_in)

            loss=(cur_res-cur_lab)**2
            loss=torch.mean(loss)

            avg_train_loss=avg_train_loss+loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss=avg_train_loss/num_batches

        if epoch%eval_epoch==0:
            mapper.save(epoch)
            if device=='cuda':
                avg_train_loss=avg_train_loss.cpu()
            print('train -- epoch: '+str(epoch)+' loss: '+str(avg_train_loss.detach().numpy())+'\n')

            avg_val_loss=0
            num_batches_val=int(val_data.shape[0]/(batch_size[0]*batch_size[1]))
            for batch_index in range(num_batches_val):
                cur_in,cur_lab=get_batch(val_data,batch_size)
                cur_in=torch.tensor(cur_in).to(device).float()
                cur_lab=torch.tensor(cur_lab).to(device).float()
                cur_res=mapper.forward(cur_in)

                loss=(cur_res-cur_lab)**2
                loss=torch.mean(loss)

                avg_val_loss=avg_val_loss+loss
            avg_val_loss=avg_val_loss/num_batches_val
            if device=='cuda':
                avg_val_loss=avg_val_loss.cpu()
            print('val -- epoch: '+str(epoch)+' loss: '+str(avg_val_loss.detach().numpy())+'\n')



if __name__=='__main__':
    # mapper=PolyMapper(3)
    # input=torch.rand([1,3,5,6])
    # res=mapper.forward(input)

    # mapper=PolyMapper(3)
    # input='./color mapping/1 - 2023-7-6 - selected/p1.jpeg'
    # mapper.inspect_patch(input)


    degree=2
    name='2023_7_13_v1'
    device='cpu'
    load=False

    # mapper=PolyMapper(degree,name=name,device=device,load=load)
    # train_test(mapper)
    # input='./color mapping/1 - 2023-7-6 - selected/p1.jpeg'
    # mapper.inspect_patch(input)

    load=True
    mapper=PolyMapper(degree,name=name,device=device,load=load)
    input='./color mapping/1 - 2023-7-6 - selected/p16.jpeg'
    mapper.inspect_patch(input)

