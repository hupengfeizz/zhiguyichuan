# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/20 10:30
"""


''' 读取 yolo 标签，转 voc 格式，并输出 xml 文件 
    step1. 读取整个目录的文件，for 读一个文件
           单独处理类别文件 classes.txt 保存字典 {0:类别名}
    step2. 读取 yolo 数据（一个标签文件，考虑有多个类别，使用 for 循环）
            数据最类归一化：img 的 size
    step3. 根据文件名，拼接图片文件的路径，导入 PIL 模块，获取 img 的size
    step4. 转 voc，【class_id, xmin, ymin, xmax, ymax】
    step5. 生成 xml文件，path、size、name、bbox
'''


import os
import PIL.Image as Image
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


def read_yolo_to_voc(yolo_label_file_path, img_width, img_height):
    voc_box = []
    with open(yolo_label_file_path, mode='r') as f:
        for line in f.readlines():
            class_id, x, y, w, h = line.strip().split()
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)

            xmin = int((x - w / 2) * img_width)
            ymin = int((y - h / 2) * img_height)
            xmax = int((x + w / 2) * img_width)
            ymax = int((y + h / 2) * img_height)

            voc_box.append((class_id, xmin, ymin, xmax, ymax))
    return voc_box


def generate_xml(xml_file_path, voc_boxs, img_width, img_height, image_path):
    # 构造根节点 annotation
    root = ET.Element('annotation根节点')
    # 构造子节点 path
    ET.SubElement(root, 'path').text = image_path
    # 构造子节点 size
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)

    # # 构造标签数据（多个对象）：调用 read_yolo_to_voc 返回值
    # voc_boxs = read_yolo_to_voc(voc_box, img_width, img_height)
    # # [(0, 70, 25, 145, 109), (1, 81, 68, 140, 109)]
    for box in voc_boxs:
        class_id, xmin, ymin, xmax, ymax = box

        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = class_dict[class_id]
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # 创建文件写文件，
    # ET.ElementTree(root).write(xml_file_path, encoding='utf-8')
    # 格式化写入 xml
    # format_xml = parseString(ET.tostring(root)).toprettyxml(indent="\t")
    with open(xml_file_path, mode='w', encoding='utf-8') as f:
        f.write(parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="   "))
    # pass

    print('完成')


if __name__ == '__main__':
    # 使用相对路径，指定标签目录
    yolo_label_dir = './scipts/label/label_yolo'

    # 图片的目录
    image_dir = './scipts/label/images'

    new_xml_dir = './scipts/new_xml_dir'
    # 创建 xml 文件的目录，保存生成 xml 文件
    os.makedirs(new_xml_dir, exist_ok=True)

    # 拼接 classes.txt 文件的路径
    class_name_path = os.path.join(yolo_label_dir, 'classes.txt')

    # 打开文件 classes.txt，获取类别名，保存到字典中
    with open(class_name_path, mode='r') as f:
        class_dict = {idx:name.strip() for idx,name in enumerate(f.readlines())}
    print(class_dict)

    # 获取 yolo_label_dir 目录下的所有的文件，判断是 txt，并过滤 classes.txt
    files = [f for f in os.listdir(yolo_label_dir) if f.endswith('.txt') and f != 'classes.txt']
    print(files)

    # # 获取 image_dir 目录下的所有的图片，判断是 jpg
    # images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    # print(images)

    # 循环读取每个标签文件名
    for file in files:
        # 拼接文件的路径
        file_path = os.path.join(yolo_label_dir, file)

        # 根据标签文件名，找对应的图片
        file_name, _ = os.path.splitext(file)

        # 拼接图片的路径
        image_name = os.path.join(image_dir, file_name + '.jpg')

        # yolo 数据进行归一化，获取 img 大小
        img_width, img_height = Image.open(image_name).size

        # 调用封装的函数
        voc_boxs = read_yolo_to_voc(file_path, img_width, img_height)

        # 拼接保存的 xml 文件的路径（新目录, file_name + '.xml'）
        xml_file_path = os.path.join(new_xml_dir, file_name + '.xml')

        # 调用函数 generate_xml
        generate_xml(xml_file_path, voc_boxs, img_width, img_height, image_name)

    pass