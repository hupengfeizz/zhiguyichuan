# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/20 16:09
"""
# from PyQt5.QtCore.QUrl import toLocalFile
from imgviz.data import voc

''' coco 数据集，返回的文件格式是 json 格式 '''

''' json 格式与 dict 一样，{"key":"value"} 
    不同的是，json 中的键必须是字符串，而且必须使用双引号
'''

import json

''' json 函数：
1. load 读 json 文件, loads
   json.load(文件对象)          # 适合大文件
   json.loads(字符串)           # 适合数据量小
'''
# json 里面数据：x-(左上xmin), y(左上ymin), withd(box的宽), height(box的高)

json_file_path = r'./scipts/label/label_coco/test_00001546.json'

# 读取 json 文件的内容：打开文件，读取内容
with open(json_file_path, 'r', encoding='utf-8') as f:
    # load 读取 f 文件对象，返回数据，字典列表
    # data = json.load(f)
    # loads 读取字符串（文件内容）
    # s=f.read()
    data = json.loads(f.read())
    print(data)

    coco_box = []
    label_boxs = data[0]['annotations']
    for label in label_boxs:
        class_name = label['label']
        # x = label['coordinates']['x']
        # y = label['coordinates']['y']
        # w = label['coordinates']['width']
        # h = label['coordinates']['height']
        x,y,w,h = [label['coordinates'][k] for k in ['x','y','width','height']]
        coco_box.append([class_name, x, y, w, h])
    print(coco_box)


''' 
2. dump 写 json 文件, dumps
   json.dump(对象，写入文件对象, indent=4)       # 直接将对象写入文件对象
   json.dumps(对象, indent=4))                # 将对象转为 json 格式的字符串，文件对象.write()
'''

d = [{"image":"图片名称",
      "annotations":[{"label":{"x":100,"y":"左上y","width":"box的宽","height":"box的高"}}]}]

with open('new_json1.json', mode='w', encoding='utf-8') as f:
    # json.demp(将对象写入文件对象中)
    json.dump(d, f, ensure_ascii=False, indent=4)
    print('1')

    # f.write (json.dumps(对象))
    # f.write(json.dumps(d, indent=4, ensure_ascii=False))





'''
voc : xmin, ymin, xmax, ymax        xmin, ymin 左上， xmax, ymax 右下
yolo：x, y, w, h                     x,y 框的中心点, w,h 框宽高（数据进行归一化）
coco：x, y, width, height            x,y 框的左上坐标，width, height 框的宽高
'''
"""
任务：完善 voc、yolo、coco 互转的
voc -> yolo 
voc -> coco 

yolo -> voc 
yolo -> coco 

coco -> voc 
coco -> yolo
"""