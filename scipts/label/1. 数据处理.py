# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/21 14:12
"""

''' 数据处理时，先备份数据 '''

import shutil
import uuid
import os

'''
    shutil 对文件和目录进行拷贝、复制
    uuid 生成唯一标识    '''

''' 1） shutil.copy('源文件'，'目标文件 或 目标目录')
        shutil.copy2('源文件'，'目标文件 或 目标目录')    保留文件的创建时间
'''
src_file = './lenna.png'
tar_file = 'lenna_copy.png'
shutil.copy(src_file, tar_file)           # 目标文件存在，覆盖
shutil.copy2(src_file, tar_file)          # 保留创建时间
shutil.copy(src_file, r'./coco128')

''' shutil.copyfile(’源文件'，'目标文件)  '''
src_image = r'./coco128/lenna.png'
tar_image = r'./coco128/lenna_copy.png'
shutil.copyfile(src_image, tar_image)


''' 2) shutil.copytree('源目录', '目标目录')  
    目标目录存在报错
'''
src_dir = './coco128'
tar_dir = './coco128_copy'
if not os.path.exists(tar_dir):
    shutil.copytree(src_dir, tar_dir)



''' 3) shutil.rmtree(移除目录，ignore_errors+True)'''
# shutil.rmtree(tar_dir, ignore_errors=True)



''' 4) shutil.move() 移除 '''
src_image1 = r'./coco128/lenna.png'
# tar_image1 = r'./coco128_copy/lenna.jpg'
shutil.move(src_image1, r'./coco128/lenna.jpg')


''' 5) uuid.uuid4() '''
uid = uuid.uuid4()          # 888bbc25-d368-4f55-92e1-29e9a8c63a09
print(type(uid))
print(str(uid).replace('-',''))


