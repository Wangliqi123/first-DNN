import numpy as np
import random

# import matplotlib as mpl
# import matplotlib.pyplot as plt

import pickle
import gzip

def load_data():
    f = gzip.open('I:/Administrator/PycharmProjects/mnist_data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='iso-8859-1')
    f.close()
    # print(type(training_data))
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results)) #把training_data的标签变成one-hot
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class Network(object):  # 建立网络类
    def __init__(self, sizes):  # 定义实例网络的各参数 sizes=[784,20,10]表示输入784维，一层隐藏层20个神经元
      self.num_layers = len(sizes)
      self.sizes = sizes
      self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
      self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])] # size[:-1]=[784,20] size[1:]=[20.10]


    def sigmoid(self,z): # 激活函数，z向量
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z): #激活函数倒导数
        return self.sigmoid(z) * (1- self.sigmoid(z))


    def feedforward(self, a): # 向前传播，激活函数sigmoid，a是初始输入，由for循环得到最后输出，验证集用
      for b, w in zip(self.biases, self.weights):
          a = self.sigmoid((np.dot(w, a)+b))
      return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): # eta学习率
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data) # 将序列所有元素随机排序
            mini_batches = [training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)] # 把训练集分割成小样本batches
            for mini_batch in mini_batches:
                 self.update_mini_batch(mini_batch, eta)  # 梯度法在一个batch上更新参数
            if test_data:
                  print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
            else:
                 print("Epoch {0} complete".format(j+1))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]# 参数变化量设置为0
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # 每一个训练样本的参数变化量
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 参数量累加
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw  # 下降梯度取这个batch的平均梯度
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors(中间变量，不加激活函数的), layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
            # 当a，b类型是numpy.ndarray，a*b表示hadamard乘积。矩阵乘积要用numpy.matrix类型
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # 矩阵置换transpose
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): # 评估
        test_results = [(np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y): # 损失函数的第一步delta误差梯度，这里取1/2的l2误差
        return (output_activations-y)


training_data, validation_data, test_data = load_data_wrapper()

# net = Network([784, 30, 10])

net = Network([784, 30, 10])

net.SGD(training_data, 3, 10, 1.0, test_data=test_data)  #30 次迭代期，⼩批量数据⼤⼩为 10，学习速率 η = 1.0