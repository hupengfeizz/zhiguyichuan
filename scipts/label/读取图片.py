# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/20 11:15
"""

# 第三方库：pip install pillow

import PIL.Image as Image

img_path = r'/Users/yoyo/AI36/data_pro/lenna.png'

img = Image.open(img_path)
print(img.size)
print(img.width)
print(img.height)
# img.show()


img1 = img.resize((100, 100))
img2 = img.resize((1000, 1000))
img3 = img.resize((512, 512), box=(100, 100, 300, 300))

img3.save('100*300.png')