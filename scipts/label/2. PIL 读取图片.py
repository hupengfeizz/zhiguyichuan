# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/20 11:15
"""

# 第三方库：pip install pillow

import PIL.Image as Image

img_path = r'/Users/yoyo/AI36/data_pro/lenna.png'
''' 1. open 打开图片，获取属性'''
img = Image.open(img_path)
print(img.size)
print(img.width)
print(img.height)
# img.show()


''' 2. resize 裁剪图片 '''
img1 = img.resize((100, 100))
img2 = img.resize((1000, 1000))
img3 = img.resize((512, 512), box=(100, 100, 300, 300))


''' 3. save 保存图片'''
img3.save('100*300.png')


''' 4. convert 改图片的模式：L灰度图，RGB彩图 '''
img4 = Image.open('lenna.png')
''' 修改 lenna.png 图片的扩展名为 jpg，不能直接修改的 '''
img4.convert('RGB').save('lenna.jpg')
img4.convert('L').save('lenna1.png')


''' 5. rotate 旋转图片 '''
img5 = img.convert('L')
img6 = img4.rotate(90)
img7 = img4.rotate(180) 
img8 = img4.rotate(270)


# 做一个动态图
# 序列 = [图片1，图片2，图片3，。。。。]
''' save(保存路径，quality=5, save_all=True, append_images=序列, loop=0, duration=毫秒) '''

lst = [img4, img6, img8, img8]
lst[0].save('动图.gif', save_all=True, append_images=[lst[1], lst[2], lst[3]], loop=0, duration=1000)

