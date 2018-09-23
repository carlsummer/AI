import numpy as np


class Perceptron(object):
    """
    eta :学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter
        pass

    """
    对神经分叉所得出的权重进行点积
    """
    def net_input(self, X):
        """
        z = W0*1 + W1*X1 + ..... Wn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    """
    神经末梢进行判断
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1);
        pass

    """对输入的样本进行神经元的培训fit"""
    def fit(self, X, y):
        """
        输入训练数据，培训神经元，x输入样本向量，y对应样本分类

        X:shape[n_samples,n_features]
        X:[[1,2,3],[4,5,6]]
        n_samples:2  总共有多少个输入的样本
        n_features:3 总共有多少个输入的电信号

        y:[1,-1]  样本分类有2个
        """

        """
        初始化权重向量为0
        np.zeros(5)
        array([ 0.,  0.,  0.,  0.,  0.])
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(x,y)=[([1,2,3],1)]
            """
            for xi, target in zip(X, y):
                """
                update = n + （y - y'）
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update * xi等价：
                [∨w(1) = x[1]*update,∨w(2) = x[2]*update,∨w(3) = x[3]*update]
                self.w_[1:] 让w_向量从序数为1开始忽略序数为0的元素
                """
                self.w_[1:] += update * xi
                self.w_[0] = update

                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass

            pass


        pass