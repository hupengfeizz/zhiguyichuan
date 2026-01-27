# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/21 15:06
"""


''' 1. 统一图片扩展名的脚本 png 或 jpg '''
import PIL.Image as Image
import os

def img2jpg(image_path, output_path):
    ''' 读取图片，转模式 '''
    img = Image.open(image_path)
    img.convert('RGB').save(output_path)


if __name__ == '__main__':
    # 源目录
    src_img_dir = 'pict'
    # 目标目录，创建目录
    target_dir = './output'
    os.makedirs(target_dir, exist_ok=True)

    all_image = os.listdir(src_img_dir)
    for img_name in all_image:
        # 拼接源图片路径
        img_path = os.path.join(src_img_dir, img_name)
        # 拼接目标图片的路径
        output_path = os.path.join(target_dir, os.path.splitext(img_name)[0] + '.jpg')
        # 调用函数处理
        img2jpg(img_path, output_path)

