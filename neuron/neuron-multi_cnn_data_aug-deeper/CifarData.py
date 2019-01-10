import os
import pickle
import numpy as np

CIFAR_DIR = "../cifar-10-batches-py"


def load_data(filename):
    """read data from data file."""
    with open(os.path.join(CIFAR_DIR, filename), "rb") as f:
        data = pickle.load(f, encoding='latin1')
        return data['data'], data['labels']


class CifarData:
    # shuffle洗牌使得数据间更加没有规律
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)  # 从纵向将所有的数据合并到一起
        self._data = self._data / 127.5 - 1  # 对数据进行归1化 self._data是0-255 除以127.5 那么就是0-2 直接的数 -1 那么就是 0-1之间的数了
        self._labels = np.hstack(all_labels)  # 从横向将所有的数据合并到一起
        print(self._data.shape)
        print(self._labels.shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0  # 已经将数据集遍历到那个位置了 指示器
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1] 将数据集混排 permutation置换
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            # raise 提升
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join('data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join("test_batch")]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

# batch_data, batch_labels = train_data.next_batch(10)
# print(batch_data)
# print(batch_labels)
