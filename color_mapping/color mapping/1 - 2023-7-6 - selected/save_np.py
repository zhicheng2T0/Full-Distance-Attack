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


file_name='inspect.JPG'
input = Image.open(file_name)
input = np.asarray(input)
input = torch.tensor(input)
input=torch.transpose(input,1,2)
input=torch.transpose(input,0,1)
resize_transform = T.Resize(size = (300,300))#.to(cfg.device)
input=resize_transform(input)
input=torch.rot90(input,2,[2,3])
nput=torch.transpose(input,0,1)
input=torch.transpose(input,1,2)
input=torch.squeeze(input)
input=input.numpy()
input=np.reshape(input,[1,3,300,300])
np.save('inspect_2023_7_13.npy',input)
