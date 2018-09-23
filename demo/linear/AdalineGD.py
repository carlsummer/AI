import numpy as np

class AdalineGD(object):
    """
    eta:float学习效率，处于0和1
    n_iter:int对训练数据进行学习改进次数
    w_:一维向量存储权重数值
    error_:存储每次迭代改进时，网络对数据进行错误判断的次数
    """
    def __init__(self,eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    """
    对神经分叉所得出的权重进行点积
    """
    def net_input(self, X):
        """
        z = W0*1 + W1*X1 + ..... Wn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def fit(self,X,y):
        """
        :param X: 二维数组[n_samples 表示X中含有训练数据条目数,n_features含有4个数据的一维向量，用于表示一条训练条目]
        :param y: 一维向量，用于存储每一训练条目对应的正确分类
        :return:
        """
        # np.zeros(5)
        #array([ 0.,  0.,  0.,  0.,  0.])
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X); #output = w0 + w1*x1 + ...wn+xn
            errors = (y - output)
            """
           x = np.array([
                [1,2],[2,3],[3,4]
            ]) 
            print(x.T)
            [[1 2 3][2 3 4]]
            print(x.T.dot(2))
            [[2 4 6][4 6 8]]
           """
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >= 0,1,-1)