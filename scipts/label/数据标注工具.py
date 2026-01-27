# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/19 10:58
"""
import os

'''
labelimg 使用 python3.10+，需要修改下面2个文件
1. 把 labelImg.py 文件拷贝到下面目录，覆盖
   C:自己配置的环境\Lib\site-packages\labelImg\labelImg.py

2. 把 canvas.py 文件拷贝到下面目录，覆盖
   C:\\new_conda\datapy310\Lib\site-packages\libs\canvas.py 
'''

'''
AI项目：
1. 确认需求，指标（P、R）
2. 数据处理：数据标注，数据清洗（修改文件名，统一图片的后缀，过滤数据，划分数据集）
3. 确认算法模型
4. 训练、优化
5. 部署 ubuntu22.04｜ 20.04（服务端、边缘端）
6. 验证、收集真实数据，继续优化
'''

'''
labelimg 工具有三种数据格式：
    voc: 标签文件 .xml
    yolo: 标签文件 .txt
    coco: 标签文件 .json
'''

''' 介绍 xml 模块，导入模块（lxml 第三方模块） '''
import xml.etree.ElementTree as ET

xml_file = r'./label_voc/test_00002490.xml'  # 尽量使用相对路径
class_names = ['face', 'mask']

# 1. parse 解析 .xml 文件的语法，返回 element 树对象
tree = ET.parse(xml_file)

# 2. 通过 element 树对象，打点调用方法 getroot 获取 xml 树结构的根节点
root = tree.getroot()

# root = ET.parse(xml_file).getroot()
print(root)

""" 获取一个框 bbox
# 3. 从 root 开始查找子节点（size、name、bbox）
''' 相对路径：// 从当前匹配节点开始 
    绝对路径：/ 从根开始查找 '''

''' find('子节点/属性') 查找一个，返回子节点对象，通过对象.text 属性值，返回字符串 '''
size = root.find('size')
width = int(size.find('width').text)
height = int(size.find('height').text)

''' findtext('子节点/属性') 查找一个，返回属性的值，返回字符串类型 '''
class_name = root.findtext('object/name')

# 将已知 list 类别的名字，转为字典
class_name_dict = {k: idx for idx, k in enumerate(class_names)}
# print(class_name_dict)
class_id = class_name_dict[class_name]

bbox = root.find('object/bndbox')
xmin = int(bbox.findtext('xmin'))
ymin = int(bbox.findtext('ymin'))
xmax = int(bbox.find('xmax').text)
ymax = int(bbox.find('ymax').text)

bbox = (class_id, xmin, ymin, xmax, ymax)
print(bbox)
"""


""" 获取多个对象 bbox """
class_name_dict = {k: idx for idx, k in enumerate(class_names)}
''' findall('子节点') 获取多个对象，返回列表【元素树对象1, 元素树对象2,】 '''
objects = root.findall('object')
bbox = []
for obj in objects:
    class_name = obj.findtext('name')
    if class_name in class_name_dict:
        class_id = class_name_dict[class_name]

    xmin = int(obj.findtext('bndbox/xmin'))
    ymin = int(obj.findtext('bndbox/ymin'))
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)

    bbox.append((class_id, xmin, ymin, xmax, ymax))

print(bbox)
pass















