import json

# Reading from file
with open('/home/hulab/zhicheng/codes/adversarial_cloth/data/INRIAPerson/VOC2007_COCO/annotations/instances_val2017.json', 'r+') as f:
  data = json.load(f)
  for i in range(len(data['images'])):
    file_name=data['images'][i]['file_name']
    newfn=file_name[0:len(file_name)-4]+'.png'
    data['images'][i]['file_name']=newfn
    print(data['images'][i])
  f.seek(0)        # <--- should reset file position to the beginning.
  json.dump(data, f, indent=4)
  f.truncate()


# Reading from file
with open('/home/hulab/zhicheng/codes/adversarial_cloth/data/INRIAPerson/VOC2007_COCO/annotations/instances_train2017.json', 'r+') as f:
  data = json.load(f)
  for i in range(len(data['images'])):
    file_name=data['images'][i]['file_name']
    newfn=file_name[0:len(file_name)-4]+'.png'
    data['images'][i]['file_name']=newfn
    print(data['images'][i])
  f.seek(0)        # <--- should reset file position to the beginning.
  json.dump(data, f, indent=4)
  f.truncate()


# Reading from file
with open('/home/hulab/zhicheng/codes/adversarial_cloth/data/INRIAPerson/VOC2007_COCO/annotations/instances_test2017.json', 'r+') as f:
  data = json.load(f)
  for i in range(len(data['images'])):
    file_name=data['images'][i]['file_name']
    newfn=file_name[0:len(file_name)-4]+'.png'
    data['images'][i]['file_name']=newfn
    print(data['images'][i])
  f.seek(0)        # <--- should reset file position to the beginning.
  json.dump(data, f, indent=4)
  f.truncate()