import matplotlib.pyplot as  plt
import main2 as mm
import demo.common.dataReady as  dataReady
from common import plotDec

ppn = mm.Perceptron(eta = 0.1,n_iter=50)
ppn.fit(dataReady.X,dataReady.y)
plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
plt.show()

plotDec.plot_decision_regions(dataReady.X,dataReady.y,ppn,resolution=0.02)
plt.xlabel(u'花瓣长度')
plt.ylabel(u'花径长度')
plt.legend(loc='upper left')
plt.show()

