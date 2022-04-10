import numpy as np
import struct
import os
from tensorboardX import SummaryWriter

MNIST_DIR = r"C:\Users\lidem\Desktop\神经网络和深度学习"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"


np.random.seed(45)
EPOCH = 50
BATCHSIZE = 128


def load_mnist(file_dir, is_images='True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data


def load_data():
    # 调用函数 load_mnist 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
    print('Loading MNIST data from local files...')
    train_images = load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
    test_images = load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
    train_data = np.append(train_images, train_labels, axis=1)
    test_data = np.append(test_images, test_labels, axis=1)
    return train_data, test_data


def shuffle_data(data):
    print('Randomly shuffle MNIST data...')
    np.random.shuffle(data)


def cal_accuracy(y, y_pred):
    """calculate accuracy"""
    p = np.argmax(y, axis=0)  # [1, N]
    q = np.argmax(y_pred, axis=0)
    acc = np.sum(p == q) / y.shape[1]
    return acc


# 激活函数
class Relu:
    def __init__(self):
        self.result = None

    def forward(self, x):
        self.x = x
        self.result = np.maximum(0, x)
        return self.result

    def backward(self, grad):
        new_grad = grad
        new_grad[self.x < 0] = 0
        return new_grad


class Sigmoid:
    def __init__(self):
        self.result = None

    def forward(self, x):
        self.result = 1 / (1 + np.exp(-x))
        return self.result

    def backward(self, grad):
        return grad * self.result * (1 - self.result)


class SoftmaxLossLayer:
    def __init__(self):
        self.label = None

    def forward(self, x):
        input_max = np.max(x, axis=1, keepdims=True)
        input_exp = np.exp(x - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        grad = (self.prob - self.label_onehot) / self.batch_size
        return grad


# 定义全连接网络
class Linear:
    def __init__(self, in_chan, out_chan, alpha):
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.alpha = alpha

    def init_param(self, std=0.01):
        self.W = np.random.normal(loc=0.0, scale=std, size=(self.in_chan, self.out_chan))
        self.b = np.zeros([1, self.out_chan])

    def forward(self, x):
        # import pdb;pdb.set_trace()
        self.inputs = x
        self.outputs = np.matmul(self.inputs, self.W) + self.b
        return self.outputs

    def backward(self, grad):
        cur_grad = np.dot(grad, self.W.T)
        self.d_W = np.dot(self.inputs.T, grad) + self.alpha * self.W
        self.d_b = np.sum(grad, axis=0)
        return cur_grad

    def updata(self, lr):
        self.W -= lr * self.d_W
        self.b -= lr * self.d_b

    def save_param(self):
        return self.W, self.b

    def load_param(self, W, b):
        assert self.W.shape == W.shape
        assert self.b.shape == b.shape
        self.W = W
        self.b = b


class MNIST_MLP:
    def __init__(self, alpha, input_size=784, hidden=256, out_class=10):

        self.f1 = Linear(input_size, hidden, alpha)
        self.activate = Sigmoid()
        self.f2 = Linear(hidden, out_class, alpha)
        self.softmax = SoftmaxLossLayer()

    def init_MLP(self):
        self.f1.init_param()
        self.f2.init_param()

    def forward(self, x):
        h1 = self.f1.forward(x)
        h1 = self.activate.forward(h1)
        h2 = self.f2.forward(h1)
        prob = self.softmax.forward(h2)
        return prob

    def backward(self):
        d_loss = self.softmax.backward()
        grad = self.f2.backward(d_loss)
        grad = self.activate.backward(grad)
        grad = self.f1.backward(grad)

    def updata(self, lr):
        self.f1.updata(lr)
        self.f2.updata(lr)

    def save_model(self, param_dir):
        print("Save model to file "+param_dir)
        param = {}
        param['W1'], param['b1'] = self.f1.save_param()
        param['W2'], param['b2'] = self.f2.save_param()
        np.save(param_dir, param)

    def load_model(self, param_dir):
        print("loading model from file "+param_dir)
        param = np.load(param_dir, allow_pickle=True).item()
        self.f1.load_param(param['W1'], param['b1'])
        self.f2.load_param(param['W2'], param['b2'])

    def get_loss(self, label):
        return self.softmax.get_loss(label)


train_data, test_data = load_data()
N = len(train_data)

writer = SummaryWriter()


def train():
    print('Start training...')
    # lr_scale = [0.001, 0.005, 0.007, 0.01, 0.05]
    lr_scale = [0.01]
    # hidden_num = [64, 128, 256]
    hidden_num = [256]
    # alpha_scale = [0, 0.0001, 0.0003, 0.001]
    alpha_scale = [0.0001]
    best_lr, best_hidden, best_alpha, best_acc, best_epoch = 0,0,0,0,0
    max_batch = train_data.shape[0] // BATCHSIZE
    for lr in lr_scale:
        for hidden in hidden_num:
            for alpha in alpha_scale:
                net = MNIST_MLP(alpha, 784, hidden, 10)
                net.init_MLP()
                for epoch in range(EPOCH):
                    # shuffle_data(train_data)
                    for idx_batch in range(max_batch):
                        images = train_data[idx_batch*BATCHSIZE : (idx_batch+1)*BATCHSIZE, :-1]
                        labels = train_data[idx_batch*BATCHSIZE : (idx_batch+1)*BATCHSIZE, -1]
                        # import pdb;pdb.set_trace()
                        prob = net.forward(images)
                        loss = net.get_loss(labels)
                        net.backward()
                        net.updata(lr)

                    writer.add_scalar('lr', lr, global_step=epoch)

                    lr -= lr/100
                    print('Epoch %d, loss: %.6f' % (epoch, loss))
                    y_prob = net.forward(train_data[:,:-1])
                    train_loss = net.get_loss(train_data[:, -1])
                    y_pred = np.argmax(y_prob, axis=1)
                    acc = np.mean(y_pred == train_data[:, -1])
                    # acc = cal_accuracy(y_prob, train_data[:,-1])
                    print('train accuracy: ', acc)
                    writer.add_scalar('train_loss', train_loss, global_step=epoch)

                    y_test_prob = net.forward(test_data[:,:-1])
                    test_loss = net.get_loss(test_data[:, -1])
                    y_test_pred = np.argmax(y_test_prob, axis=1)
                    test_acc = np.mean(y_test_pred == test_data[:, -1])
                    # test_acc = cal_accuracy(y_test_prob, test_data[:,-1])
                    print('test accuracy: ', test_acc)
                    writer.add_scalar('test_loss', test_loss, global_step=epoch)
                    writer.add_scalar('test_accuracy', test_acc, global_step=epoch)

                    if test_acc > best_acc:
                        best_acc, best_lr, best_hidden, best_alpha, best_epoch = test_acc, lr, hidden, alpha, epoch
                        net.save_model('2level.npy')
    print('best_lr: ', best_lr, 'best_hidden: ', best_hidden, 'best_alpha: ', best_alpha, 'best_epoch: ', best_epoch,' best_acc: ', best_acc)


if __name__ == '__main__':
    train()




