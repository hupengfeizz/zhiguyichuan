# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/19 15:59
"""

# 做口罩识别：类别 mask，face
# bbox = [class_id, xmin, ymin, xmax, ymax]

''' 第一部分：导入需要模块 '''
import os
import xml.etree.ElementTree as ET


''' 第二部分：封装函数、类 '''
# 函数处理 xml 文件
def read_voc(xml_file_path):
    '''
    函数作用：读取 xml 文件，返回 bbox
    :param xml_file_path:  xml文件路径
    :param class_names:  类别名称
    :return:  bbox = 【class_id, xmin, ymin, xmax, ymax】
    '''
    # 把类别转为字典格式（给每个类别编号）：便于读取
    class_name_dict = {name:idx for idx, name in enumerate(class_names)}

    # 1. 解析 xml 文件，返回 element 树对象 tree
    tree = ET.parse(xml_file_path)
    # 2. 从 element 树对象 tree 中，获取根节点 root
    root = tree.getroot()
    # 3. 获取 类别 name 以及对应的 box（考虑可能多个 object）
    voc_bbox = []
    objects = root.findall('object')
    # 循环从 objects 列表，读取每个对象 obj
    for obj in objects:
        # 获取类别的名称，字符串
        class_name = obj.findtext('name')
        if class_name not in class_name_dict:
            raise '类别不存在'
        # 通过类别字典，获取类别编号
        class_id = class_name_dict[class_name]
        xmin = int(obj.findtext('bndbox/xmin'))
        ymin = int(obj.findtext('bndbox/ymin'))
        xmax = int(obj.findtext('bndbox/xmax'))
        ymax = int(obj.findtext('bndbox/ymax'))

        voc_bbox.append([class_id, xmin, ymin, xmax, ymax])
    return voc_bbox


def read_voc_to_yolo(xml_file_path):
    '''
    函数作用：读取 xml 文件，返回 bbox
    :param xml_file_path:  xml文件路径
    :param class_names:  类别名称
    :return:  bbox = 【class_id, xmin, ymin, xmax, ymax】
    '''
    # 把类别转为字典格式（给每个类别编号）：便于读取
    class_name_dict = {name:idx for idx, name in enumerate(class_names)}

    # 1. 解析 xml 文件，返回 element 树对象 tree
    tree = ET.parse(xml_file_path)
    # 2. 从 element 树对象 tree 中，获取根节点 root
    root = tree.getroot()
    # 3. 获取图片的 size
    width = int(root.findtext('size/width'))
    height = int(root.findtext('size/height'))
    # 4. 获取 类别 name 以及对应的 box（考虑可能多个 object）
    voc_bbox = []
    objects = root.findall('object')
    # 循环从 objects 列表，读取每个对象 obj
    for obj in objects:
        # 获取类别的名称，字符串
        class_name = obj.findtext('name')
        if class_name not in class_name_dict:
            raise '类别不存在'
        # 通过类别字典，获取类别编号
        class_id = class_name_dict[class_name]
        xmin = int(obj.findtext('bndbox/xmin'))
        ymin = int(obj.findtext('bndbox/ymin'))
        xmax = int(obj.findtext('bndbox/xmax'))
        ymax = int(obj.findtext('bndbox/ymax'))

        # 转 yolo 格式
        x = ((xmax - xmin) / 2 + xmin) / width
        y = ((ymax - ymin) / 2 + ymin) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        voc_bbox.append([class_id, x, y, w, h])
    return voc_bbox


''' 第三部分：主程序 main '''
if __name__ == '__main__':
    # 已知类别
    class_names = ['mask', 'face']

    label_voc_dir = 'scipts/label/label_voc'

    # 获取目录下的所有 xml 文件名, 列表
    all_file_name = os.listdir(label_voc_dir)

    for file_name in all_file_name:
        # 拼接 xml 文件的路径
        xml_file = os.path.join(label_voc_dir, file_name)
        # print(read_voc(xml_file))
        print(read_voc_to_yolo(xml_file))

