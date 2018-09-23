import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#设置中文字体
matplotlib.use('qt4agg')
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'

print(int(False))
file = "./iris.data.csv"
#header=None告诉python第一行是真实的数据
df = pd.read_csv(file,header=None)
print(df.head(10))
#获取csv中的第0行到100行的第5列的值
y = df.loc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
#获取csv中的第0行到100行的第0,2列的值
X = df.loc[0:100,[0,2]].values
#将前50行的第一列作为横坐标，第二列作为纵坐标，颜色为红色，标记为o，label为setosa
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel(u'花瓣长度')
plt.ylabel(u'花径长度')
#将标记放置左上角
plt.legend(loc='upper left')
#显示数据图
plt.show()
print(y)