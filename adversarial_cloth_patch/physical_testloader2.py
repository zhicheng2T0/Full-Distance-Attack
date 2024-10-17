import numpy as np
import os
import random
from PIL import Image
import torchvision
import torch
import torchvision.transforms as T


class PhysicalLoader:
    def __init__(self, base_dir):
        self.base_dir=base_dir

        self.normalize_trans=torchvision.transforms.Normalize(
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375])

    def get_all(self,folder,sub_folder):
        folder_dir=self.base_dir+'/'+folder
        data_dir=[]
        files=os.listdir(folder_dir+'/'+sub_folder)
        for f in range(len(files)):
            data_dir.append([folder_dir+'/'+sub_folder+'/'+files[f],folder,sub_folder,files[f]])


        image_list=[]
        for i in range(len(data_dir)):

            #get raw image
            image = Image.open(data_dir[i][0])
            raw_img = np.asarray(image)

            #process image
            #convert to tensor
            raw_img=torch.tensor(raw_img,dtype=torch.float64)
            #resize
            raw_img=torch.transpose(raw_img,2,0)
            raw_img=torch.transpose(raw_img,2,1)
            raw_img=torch.unsqueeze(raw_img,0)
            #print(raw_img.shape)
            h=raw_img.shape[2]
            w=raw_img.shape[3]
            if w<h:
                if w>800:
                    ratio=800/w
                    new_h=int(h*ratio)
                    new_w=int(w*ratio)
                    resize_transform = T.Resize(size = (new_h,new_w))
                    raw_img=resize_transform(raw_img)

            elif w>=h:
                if h>800:
                    ratio=800/h
                    new_h=int(h*ratio)
                    new_w=int(w*ratio)
                    resize_transform = T.Resize(size = (new_h,new_w))
                    raw_img=resize_transform(raw_img)
            #normalize

            #resize_transform = T.Resize(size = (416,416))
            #raw_img=resize_transform(raw_img)

            img=self.normalize_trans(raw_img)

            image_list.append([img.float(),data_dir[i][1],data_dir[i][2],data_dir[i][3]])

        return image_list


if __name__=='__main__':
    base_dir='.'
    dataloader=PhysicalLoader(base_dir)
    img_list=dataloader.get_all('adv_patch_zhi','2')
    print('break point')

