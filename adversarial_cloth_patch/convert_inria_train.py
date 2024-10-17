# coding=UTF-8

import os
import re
from PIL import Image

sets=['train']
# Variables need to be filled in image_path、annotations_path、full_path
#image_path = r"D:\BaiduNetdiskDownload\59_INRIA Person Dataset\shuju1/"
image_path = r"/content/adversarial_cloth/data/INRIAPerson/Train/"                       #  Picture storage path , Fixed path
#annotations_path = r"D:\BaiduNetdiskDownload\59_INRIA Person Dataset\INRIAPerson\Test\annotations/" # Folder Directory  # INRIA Label storage path
annotations_path = r"/content/adversarial_cloth/data/INRIAPerson/Train/annotations" # Folder Directory  # INRIA Label storage path
annotations= os.listdir(annotations_path) # Get all the file names under the folder

#  Get the picture names of all the pictures in the folder
def get_name(file_dir):
   list_file=[]
   for root, dirs, files in os.walk(file_dir):
      for file in files:
         # splitext() Split the path into file names + Extension , for example os.path.splitext(“E:/lena.jpg”) Will get ”E:/lena“+".jpg"
         if os.path.splitext(file)[1] == '.jpg':
            list_file.append(os.path.join(root, file))
   return list_file

#  stay labels Create a label for each picture under the directory txt file
def text_create(name,bnd):
   full_path = r"/content/adversarial_cloth/data/INRIAPerson/Train/labels1/%s.txt"%(name)
   size = get_size(name + '.png')
   convert_size = convert(size, bnd)
   file = open(full_path, 'a')
   file.write('0 ' + str(convert_size[0]) + ' ' + str(convert_size[1]) + ' ' + str(convert_size[2]) + ' ' + str(convert_size[3]) )
   file.write('\n')

#  Get the image to query w,h
def get_size(image_id):
   im = Image.open(r'/content/adversarial_cloth/data/INRIAPerson/Train/pos/%s'%(image_id))       #  Source image storage path
   size = im.size
   w = size[0]
   h = size[1]
   return (w,h)

#  take Tagphoto Of x,y,w,h Format to yolo Of X,Y,W,H
def convert(size, box):
   dw = 1./size[0]
   dh = 1./size[1]
   x = (box[0] + box[2])/2.0
   y = (box[1] + box[3])/2.0
   w = box[2] - box[0]
   h = box[3] - box[1]
   x = x*dw
   w = w*dw
   y = y*dh
   h = h*dh
   return (x,y,w,h)

#  Put the processed image path into a ｔｘｔ In the folder
for image_set in sets:
   if not os.path.exists(r'/content/adversarial_cloth/data/INRIAPerson/Train/labels1'):
      os.makedirs(r'/content/adversarial_cloth/data/INRIAPerson/Train/labels1')                     #  Generated yolo3 Label storage path , Fixed path
   image_names = get_name(image_path)
   list_file = open('2007_%s.txt'%(image_set), 'w')
   for image_name in image_names:
      list_file.write('%s\n'%(image_name))
   list_file.close()

s = []
for file in annotations: # Traversal folder
   str_name = file.replace('.txt', '')

   if not os.path.isdir(file): # Determine whether it is a folder , It's not a folder that opens

      with open(annotations_path+"/"+file, encoding = "ISO-8859-1") as f : # Open file
         iter_f = iter(f); # Create iterator
         for line in iter_f: # Traversal file , Traverse line by line , Read text
            str_XY = "(Xmax, Ymax)"
            #print(line)
            if str_XY in line:
               strlist = line.split(str_XY)
               strlist1 = "".join(strlist[1:])    #  hold list To str
               strlist1 = strlist1.replace(':', '')
               strlist1 = strlist1.replace('-', '')
               strlist1 = strlist1.replace('(', '')
               strlist1 = strlist1.replace(')', '')
               strlist1 = strlist1.replace(',', '')
               b = strlist1.split()
               bnd = (float(b[0]) ,float(b[1]) ,float(b[2]) ,float(b[3]))
               text_create(str_name, bnd)
            else:
               continue




