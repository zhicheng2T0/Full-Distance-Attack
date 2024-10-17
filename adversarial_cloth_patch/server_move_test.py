import shutil
from xml.dom.minidom import Document
import os
import re

list = os.listdir("pos")
print(list)


#os.mkdir( '/data/run01/scz1972/rsw_/zhicheng/adversarial_cloth/data/INRIAPerson/VOC2007_COCO/test_images/')

for i in range(len(list)):
  shutil.copy('./pos/'+list[i],'/home/hulab/zhicheng/codes/adversarial_cloth/data/INRIAPerson/VOC2007_COCO/test_images/'+list[i])