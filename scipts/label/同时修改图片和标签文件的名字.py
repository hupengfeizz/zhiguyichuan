# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/21 16:20
"""
import os
import shutil
import uuid


''' pcb_data '''
# 源图片和标签的目录
src_image_dir = './文件处理_数据集/pcb_data/images'
src_label_dir = './文件处理_数据集/pcb_data/labels'

# 创建目标的图片和标签的新目录
tar_image_dir = './文件处理_数据集/pcb_data/new_images'
tar_label_dir = './文件处理_数据集/pcb_data/new_labels'
os.makedirs(tar_image_dir, exist_ok=True)
os.makedirs(tar_label_dir, exist_ok=True)

# 创建新目录：存放没有标签的图片
not_label_dir = './文件处理_数据集/pcb_data/image_not_label'
os.makedirs(not_label_dir, exist_ok=True)

# 创建新目录：存在有标签，且标签文件的大小为0
label_size = './文件处理_数据集/pcb_data/image_label_zero'
os.makedirs(label_size, exist_ok=True)


# 修改图片名字
# 1. 读取图片
all_image = os.listdir(src_image_dir)
all_label = os.listdir(src_label_dir)
for img_name in all_image:
    # 源 img_name 图片的路径
    img_path = os.path.join(src_image_dir, img_name)
    # 生成唯一标识 uid
    uid = str(uuid.uuid1()).replace('-', '')
    # 获取图片的名 以及 后缀
    image_name, ext = os.path.splitext(img_name)
    # 01_missing_hole_01_jpg.rf.8bc03a94396d5e7d3e3f1961b3aaf308   .jpg
    # 保留原来图片名的第一个点前面的内容，修改后面的 uid
    # image_name = image_name.split('.')[0]+uid+'.jpg'
    # shutil.copy(img_path, tar_image_dir + '/' + uid + '.jpg')
     
    target_path = os.path.join(tar_image_dir, img_name)
    # 目标 的图片路径
    out_image_path = os.path.join(tar_image_dir, image_name.split('.')[0] + uid + '.jpg')

    # 注意：修改图片，标签同步也要改 uid
    # 根据你的图片名字，拼标签名字 .txt
    label_name = os.path.join(image_name + '.txt')
    if not label_name in all_label:
        # copy 图片到新目录（没有标签的图片）
        shutil.copy(img_path, not_label_dir)
    else:
        file_size = os.path.getsize(os.path.join(src_label_dir, label_name))
        if file_size != 0:
            # 根据图片名 找到 对应的标签文件，同步修改标签文件
            label_path = os.path.join(src_label_dir, label_name)

            # 标签的新目录，标签文件的新名
            out_label_path = os.path.join(tar_label_dir, label_name.split('.')[0] + uid + '.txt')

            # if 语句保证 图片和标签都存在，才执行拷贝
            shutil.copy(img_path, out_image_path)
            shutil.copy(label_path, out_label_path)
        else:
            shutil.copy(img_path, label_size)

    pass

