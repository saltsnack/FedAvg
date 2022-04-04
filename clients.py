import numpy as np
import tensorflow as tf
from dataSets import DataSet
import cv2


class user(object):
    def __init__(self, localData, localLabel, isToPreprocess):
        self.dataset = localData
        self.label = localLabel
        self.train_dataset = None
        self.train_label = None
        self.isToPreprocess = isToPreprocess

        self.dataset_size = localData.shape[0]
        self._index_in_train_epoch = 0
        self.parameters = {}

        self.train_dataset = self.dataset
        self.train_label = self.label
        if self.isToPreprocess == 1:
            self.preprocess()


    def next_batch(self, batchsize):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batchsize
        if self._index_in_train_epoch > self.dataset_size:
            order = np.arange(self.dataset_size)
            np.random.shuffle(order)
            self.train_dataset = self.dataset[order]
            self.train_label = self.label[order]
            if self.isToPreprocess == 1:
                self.preprocess()
            start = 0
            self._index_in_train_epoch = batchsize
        end = self._index_in_train_epoch
        return self.train_dataset[start:end], self.train_label[start:end]
    
    # 数据集处理
    def preprocess(self):
        new_images = []
        shape = (24, 24, 3)
        for i in range(self.dataset_size):
            old_image = self.train_dataset[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]

            if np.random.random() < 0.5:
                new_image = cv2.flip(new_image, 1)

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(self.train_dataset.shape[1] * self.train_dataset.shape[2] * self.train_dataset.shape[3])])
            new_image = (new_image - mean) / std

            new_images.append(new_image)

        self.train_dataset = new_images



class clients(object):
    def __init__(self, numOfClients, dataSetName, bLocalBatchSize,
                 eLocalEpoch, sess, train, inputsx, inputsy, is_IID):
        self.num_of_clients = numOfClients
        self.dataset_name = dataSetName
        self.dataset_size = None
        self.test_data = None
        self.test_label = None
        self.B = bLocalBatchSize
        self.E = eLocalEpoch
        self.session = sess
        self.train = train
        self.inputsx = inputsx
        self.inputsy = inputsy
        self.IID = is_IID
        self.clientsSet = {}

        self.dataset_allocation()

    # 为客户端分配随机大小的数据量
    def allocation_train_data(K, train_data_size):
        # 生成K-1个[0, train_data_size]范围内的整数
        a = []
        for i in range(K - 1):
            b = np.random.randint(0, train_data_size)
            a.append(b)

        # 向a添加0和train_data_size值
        a.append(0)
        a.append(train_data_size)

        # 将a排序
        a.sort()

        # 生成客户端数据集大小列表
        client_train_data_sizes = []
        for i in range(K):
            size = a[i + 1] - a[i]
            client_train_data_sizes.append(size)

        return client_train_data_sizes


    # 给每个客户端分配数据
    def dataset_allocation(self):
        dataset = DataSet(self.dataset_name, self.IID)
        self.dataset_size = dataset.train_data_size
        self.test_data = dataset.test_data
        self.test_label = dataset.test_label

        localDataSizes = self.allocation_train_data(self.num_of_clients, self.dataset_size)
        
        # 只需要处理CIFAR10数据集
        preprocess = 1 if self.dataset_name == 'cifar10' else 0
        
        # 给客户端分配数据集中连续的随机大小的数据
        start = end = 0
        for i in range(self.num_of_clients):
            end = end + localDataSizes[i]

            data = dataset.train_data[start: end]
            label = dataset.train_label[start: end]
            someone = user(data, label)
            self.clientsSet['client{}'.format(i)] = someone

            start = end


    # 客户端本地训练
    def ClientUpdate(self, client, global_vars):
        all_vars = tf.trainable_variables()
        for variable, value in zip(all_vars, global_vars):
            variable.load(value, self.session)

        for i in range(self.E):
            for j in range(self.clientsSet[client].dataset_size // self.B):
                train_data, train_label = self.clientsSet[client].next_batch(self.B)
                self.session.run(self.train, feed_dict={self.inputsx: train_data, self.inputsy: train_label})

        return self.session.run(tf.trainable_variables())
