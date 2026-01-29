import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(r'/Users/macbook/Documents/data/199个样本画直方图和箱线图.xlsx',header=None)#读取数据

fig,ax = plt.subplots(figsize=(6,4),facecolor='white')#创建图形
sns.histplot(df.values,bins=15,ax=ax,edgecolor='w',kde=True,label=None) #直方图
#ax.bar_label(ax.containers[0],labels=np.histogram(df.values,bins=15)[0]) #标注数据
ax.set_xlabel('Leaf nitrogen content(%)',fontsize=15) #标题
ax.set_ylabel('Frequency',fontsize=15)
ax.legend_.remove() #去除图例
ax.set_xticks(np.linspace(2,4.7,10)) #设置横轴刻度
plt.savefig('fig.jpg',dpi=600,bbox_inches='tight')
plt.show()
