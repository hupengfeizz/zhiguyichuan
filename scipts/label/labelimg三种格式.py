# -*- coding: utf-8 -*-

"""
@Author : yoyo
@Date   : 2026/1/20 09:56
"""
import xml.etree.ElementTree as ET
''' 
标注工具 labelimg : voc、yolo、coco
'''
"""
1. voc 生成是 .xml 文件
<根节点>
    <子节点>
        <属性> 属性值 </属性>
    </子节点>
</根节点>


1) root = ET.parse(xml_file_path).getroot()
2) 查找子节点：find一个对象、findall-多个对象列表中、findtext返回属性值 
"""

''' 生成 xml 文件 '''
"""
1） ET.Element('根节点')
2） ET.SubElement(根节点, '子节点').text = '属性值'
"""

root = ET.Element('annotation')
size = ET.SubElement(root, 'size')
ET.SubElement(size, 'width').text = '200'
ET.SubElement(size, 'height').text = '300'

obj = ET.SubElement(root, 'object')
ET.SubElement(obj, 'name').text = '标签名'

bbox = ET.SubElement(obj, 'bndbox')
ET.SubElement(bbox, 'xmin').text = '20'
ET.SubElement(bbox, 'ymin').text = '50'
ET.SubElement(bbox, 'xmax').text = '110'
ET.SubElement(bbox, 'ymax').text = '150'


# ET.ElementTree(root).write('构造的新的文件.xml', encoding='utf-8')

''' 格式化输出 xml 文件 '''
from xml.dom.minidom import parseString
with open('格式化xml文件.xml', mode='w', encoding='utf-8') as f:
    # ET.tostring 将构造的元素树结构转为字节码，因为有中文
    # parseSting 将字符串解析为 xml 结构，
    f.write(parseString(ET.tostring(root)).toprettyxml(indent="\t"))






