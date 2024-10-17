# -*- coding: UTF-8 -*-
from xml.dom.minidom import Document
import os
import re

list = os.listdir("annotations")
savePath = 'Annotations'
for oldfilename in list:
    if str(".txt") not in oldfilename:
        continue
    print (oldfilename)

    #raw_input(unicode('按回车键退出...','utf-8').encode('gbk'))

    fileindex = re.findall('\d+', oldfilename)

    print (fileindex)
    #raw_input(unicode('按回车键退出...','utf-8').encode('gbk'))

    print (str(int(fileindex[0])))

    #raw_input(unicode('按回车键退出...','utf-8').encode('gbk'))
    newfilename = os.path.splitext(oldfilename)[0] + ".xml"

    #print newfilename
    #raw_input(unicode('按回车键退出...','utf-8').encode('gbk'))
    f = open(os.path.join("annotations",oldfilename), "rb")
    print ('processing:' + f.name)

    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('VOC2007'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(oldfilename))
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('PASperson Database'))
    source.appendChild(database)

    annotation1 = doc.createElement('annotation')
    annotation1.appendChild(doc.createTextNode('PASperson'))
    source.appendChild(annotation1)

    fr = f.readlines()  # 调用文件的 readline()方法一次读取

    for line in fr:
        if str(line).__contains__("size"):
            sizes = []
            sizes = re.findall('\d+', line.decode('utf-8'))
            size = doc.createElement('size')
            annotation.appendChild(size)
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(sizes[0]))
            size.appendChild(width)
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(sizes[1]))
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(sizes[2]))
            size.appendChild(depth)

            segmented = doc.createElement('segmented')
            segmented.appendChild(doc.createTextNode('0'))
            annotation.appendChild(segmented)
        if (str(line).__contains__('Objects')):
            nums = re.findall('\d+', line.decode('utf-8'))
            break
    for index in range(1, int(nums[0])+1):
        for line in fr:
            if str(line).__contains__("Bounding box for object " + str(index)):
                coordinate = re.findall('\d+', line.decode('utf-8'))
                object = doc.createElement('object')
                annotation.appendChild(object)

                name = doc.createElement('name')
                name.appendChild(doc.createTextNode('person'))
                object.appendChild(name)

                pose = doc.createElement('pose')
                pose.appendChild(doc.createTextNode('Unspecified'))
                object.appendChild(pose)

                truncated = doc.createElement('truncated')
                truncated.appendChild(doc.createTextNode('0'))
                object.appendChild(truncated)

                difficult = doc.createElement('difficult')
                difficult.appendChild(doc.createTextNode('0'))
                object.appendChild(difficult)

                bndbox = doc.createElement('bndbox')
                object.appendChild(bndbox)

                #数字中包含序号，下标应从1开始
                xmin = doc.createElement('xmin')
                xmin.appendChild(doc.createTextNode(coordinate[1]))
                bndbox.appendChild(xmin)
                ymin = doc.createElement('ymin')
                ymin.appendChild(doc.createTextNode(coordinate[2]))
                bndbox.appendChild(ymin)
                xmax = doc.createElement('xmax')
                xmax.appendChild(doc.createTextNode(coordinate[3]))
                bndbox.appendChild(xmax)
                ymax = doc.createElement('ymax')
                ymax.appendChild(doc.createTextNode(coordinate[4]))
                bndbox.appendChild(ymax)
    f.close()
    f = open(os.path.join(savePath,newfilename), 'w')
    f.write(doc.toprettyxml(indent="\t"))
    f.close()
    print (str(fileindex) + " compelete")

print ('process compelete')