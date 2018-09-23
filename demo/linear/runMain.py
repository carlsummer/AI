from linear import AdalineGD as ag
from common import plotDec
import matplotlib.pyplot as plt
from common import dataReady

ada = ag.AdalineGD(eta=0.0001,n_iter=50)
ada.fit(dataReady.X,dataReady.y)
plotDec.plot_decision_regions(dataReady.X,dataReady.y, classifier=ada)
plt.title("Adaline-Gradient descent")
plt.xlabel(u'花瓣长度')
plt.ylabel(u'花径长度')
plt.legend(loc = "upper left")
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
plt.show()
