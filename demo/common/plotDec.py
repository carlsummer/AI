from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','v')
    colors=('red','blue','lightgreen','gray','cyan')
    #np.unique 去除y中的重复的
    #A = [1, 2, 2, 3, 4, 3]
    #a = np.unique(A)
    #print a
    # 输出为 [1 2 3 4]
    #ListedColormap主要用于生成非渐变的颜色映射，该方法有三个参数，分别为colors，name，N
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()

    print(x1_min,x1_max)
    print(x2_min, x2_max)

    #np.arange 生成从x1_min,到x1_max，间隔为resolution的数组
    #x = np.arange(-2,2)
    # y = np.arange(0,3)#生成一位数组，其实也就是向量
    # x
    # Out[31]: array([-2, -1,  0,  1])
    # y
    # Out[32]: array([0, 1, 2])
    # z,s = np.meshgrid(x,y)#将两个一维数组变为二维矩阵
    # z
    # Out[36]:
    # array([[-2, -1,  0,  1],
    #        [-2, -1,  0,  1],
    #        [-2, -1,  0,  1]])
    # s
    # Out[37]:
    # array([[0, 0, 0, 0],
    #        [1, 1, 1, 1],
    #        [2, 2, 2, 2]])
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min,x2_max,resolution))

    print(np.arange(x1_min,x1_max,resolution).shape)
    print(np.arange(x1_min, x1_max, resolution))
    print(xx1.shape)
    print(xx1)
    print("/*********************************/")

    #xx1.ravel() 还原为原来的单维向量[-2, -1,  0,  1]
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)

    # a = np.arange(6).reshape((3, 2))
    # array([[0, 1],
    #        [2, 3],
    #        [4, 5]])
    Z = Z.reshape(xx1.shape)
    #画分割线
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)
    #设置x,y轴的最大和最小值
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y==c1,0],y=X[y==c1,1],alpha=0.8,c=cmap(idx),
                    marker=markers[idx],label=c1)

