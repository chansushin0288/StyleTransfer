# -*- coding: utf-8 -*-

import numpy as np


def next_batch(batch_size, dat=None, x=None, y=None):

    if dat is not None:
        n, p = dat.shape
    else:
        n = x.shape[0]

    row_idx = np.random.randint(n, size=n)

    for batch_n in range(n // batch_size + 1):
        batch_idx = row_idx[batch_n*batch_size: (batch_n+1)*batch_size]
        if len(batch_idx) > 0:
            if x is not None and y is not None:
                yield x[batch_idx, ], y[batch_idx, ]
            else:
                yield dat[batch_idx, 0:p-1], dat[batch_idx, p-1]


def one_to_hot(y):
    class_set = list(set(y))
    one_to_hot_result = np.zeros((len(y), len(class_set)), dtype=np.float32)

    for idx, label in enumerate(y):
        one_to_hot_result[idx, class_set.index(label)] = 1.0

    return one_to_hot_result, class_set


def im2col(img, fh, fw, oh, ow, c, s=1):
    n, _, _, _ = img.shape
    result = np.zeros((oh * ow, n, c * fh * fw), dtype=np.float)

    i = 0
    for oh_ in range(oh):
        for ow_ in range(ow):
            result[i, :] = img[:, :, (s*oh_):(s*oh_+fh), (s*ow_):(s*ow_+fw)].reshape((n, c*fh*fw))
            i += 1

    return result.transpose((1, 0, 2)).reshape((-1, c * fh * fw))


def col2im(img, fh, fw, oh, ow, h, w, s=1):
    oh_ow_n, c_fh_fw = img.shape
    n = oh_ow_n // (oh * ow)
    c = c_fh_fw // (fh * fw)

    img = img.reshape((n, oh, ow, c, fh, fw))
    result = np.zeros((n, c, h, w))

    for oh_ in range(oh):
        for ow_ in range(ow):
            result[:, :, (s*oh_):(s*oh_+fh), (s*ow_):(s*ow_+fw)] = img[:, oh_, ow_, :, :, :]

    return result


class Layer:

    def __init__(self, n_in=None, n_out=None):
        self.size = (n_in, n_out)
        self.x, self.u = None, None
        self.param = None

    def forward(self, x, y=None):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class Input(Layer):

    def __init__(self, n_out=None, size=None):
        Layer.__init__(self, None, n_out)
        if size:
            self.size = size

    def forward(self, x, y=None):
        if len(self.size) == 2:
            return x
        else:
            n, _ = x.shape
            c, h, w = self.size
            return x.reshape((n, c, h, w))

    def backward(self, dout):
        pass


class FullyConnected(Layer):

    def __init__(self, n_in, n_out):
        Layer.__init__(self, n_in, n_out)
        self.param = {'w': 'dw', 'b': 'db'}
        self.w = np.random.normal(loc=0.0, scale=1/n_in, size=self.size)
        self.b = np.zeros(shape=(1, n_out), dtype=np.float32)
        self.dw, self.db = None, None

    def forward(self, x, y=None):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        return self.u

    def backward(self, dout):
        n = self.x.shape[0]
        self.dw = np.dot(self.x.T, dout)/n
        self.db = np.mean(dout, axis=0)
        dx = np.dot(dout, self.w.T)
        return dx


class Convolution(Layer):

    def __init__(self, img_shape, ft_shape, s=1):
        Layer.__init__(self)
        self.param = {'ft': 'dft', 'b': 'db'}
        self.C, self.H, self.W = img_shape
        self.FN, self.FH, self.FW = ft_shape
        self.n = 0
        self.S = s
        p = 0  # zero padding for convenience

        self.OH = (self.H+2*p-self.FH)//self.S + 1
        self.OW = (self.W+2*p-self.FW)//self.S + 1

        self.ft = np.random.normal(loc=0.0, scale=1/(self.C*self.FH*self.FW),
                                   size=(self.FN, self.C, self.FH, self.FW))
        self.b = np.zeros(shape=(1, self.FN), dtype=np.float32)
        self.dft, self.db = None, None

    def forward(self, x, y=None):
        """ x: (N, C, H, W)
            return: (N, FN, OH, OW)"""
        self.n, _, _, _, = x.shape
        self.x = im2col(x, self.FH, self.FW, self.OH, self.OW, self.S)
        u = np.dot(self.x, self.ft.reshape(self.FN, -1).T) + self.b
        self.u = u.reshape((self.n, self.FN, self.OH, self.OW))
        return self.u

    def backward(self, dout):
        dout = dout.reshape((self.n * self.OH * self.OW, self.FN))

        dft = np.dot(self.x.T, dout)/self.n
        self.dft = dft.T.reshape((self.FN, self.C, self.FH, self.FW))

        db = np.mean(dout, axis=0)
        self.db = db.reshape((1, self.FN))

        dx = np.dot(dout, self.ft.reshape((self.FN, -1)))
        return dx


class Pooling(Layer):

    def __init__(self, img_shape, pooling_shape):
        Layer.__init__(self)
        self.C, self.H, self.W = img_shape
        self.PH, self.PW = pooling_shape
        self.S = self.PH
        self.N = None
        p = 0
        self.OH = (self.H + 2*p - self.PH)//self.S + 1
        self.OW = (self.W + 2*p - self.PW)//self.S + 1
        self.arg_max = None

    def forward(self, x, y=None):
        self.N, _, _, _ = x.shape
        x = im2col(x, self.PH, self.PW, self.OH, self.OW, self.C, self.S)
        x = x.reshape((self.N * self.C * self.OW * self.OH, -1))

        x_ = np.max(x, axis=1, keepdims=True)
        self.arg_max = np.argmax(x, axis=1)

        x_ = x_.reshape((-1, self.C))
        return col2im(x_, 1, 1, self.OH, self.OW, self.OH, self.OW)

    def backward(self, dout):
        dout = im2col(dout, 1, 1, self.OH, self.OW, self.C)

        dout = dout.reshape((1, -1))

        tmp_row = self.N * self.C * self.OH * self.OW

        backward_dout = np.zeros((tmp_row, self.PH * self.PW))
        backward_dout[np.arange(tmp_row), self.arg_max] = dout
        backward_dout = backward_dout.reshape((-1, self.PH * self.PW * self.C))

        return col2im(backward_dout, self.PH, self.PW, self.OH, self.OW, self.H, self.W, self.S)


class Flatten(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.N, self.C, self.H, self.W = None, None, None, None

    def forward(self, x, y=None):
        """ x: (N, C, H, W)
            return: (N, W*H*C)"""
        self.N, self.C, self.H, self.W = x.shape
        return x.reshape((self.N, self.C * self.H * self.W))

    def backward(self, dout):
        return dout.reshape((self.N, self.C, self.H, self.W))


class Linear(Layer):

    def forward(self, x, y=None):
        return x

    def backward(self, dout):
        return dout


class Sigmoid(Layer):

    def forward(self, x, y=None):
        self.u = 1 / (1+np.exp(-x))
        return self.u

    def backward(self, dout):
        out = np.multiply(self.u, (1.0 - self.u))
        return np.multiply(out, dout)


class ReLU(Layer):

    def __init__(self):
        Layer.__init__(self)

    def forward(self, x, y=None):
        self.u = (x >= 0).astype(np.float)
        return self.u * x

    def backward(self, dout):
        return self.u * dout


class Softmax(Layer):

    def forward(self, x, y=None):
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_x = np.sum(x, axis=1, keepdims=True)
        self.u = x / sum_x
        return self.u

    def backward(self, dout):
        return self.u * (dout - np.sum(dout * self.u, axis=1, keepdims=True))


class L2Norm(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.z, self.y = None, None

    def forward(self, z, y=None):
        self.z, self.y = z, y
        return np.linalg.norm(self.z - self.y)**2/2

    def backward(self, dout):
        return (self.z - self.y) * dout


class CrossEntropy(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.z, self.y = None, None

    def forward(self, z, y=None):
        self.z, self.y = z, y
        return -np.sum(self.y * np.log(self.z))

    def backward(self, dout):
        return -(self.y / self.z)*dout


class Optimizer:

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, nn, tr_x,  tr_y):
        raise NotImplementedError


class SGD(Optimizer):

    def update(self, nn, tr_x, tr_y):
        nn.get_gradient(tr_x, tr_y)

        for layer in nn.get_layers():
            for param, der_param in layer.param.items():
                param_value = getattr(layer, param)
                der_value = getattr(layer, der_param)
                setattr(layer, param, param_value - self.lr * der_value)


class NeuralNetwork:

    def __init__(self, layers, loss_function, labels=None):
        self.layers = layers
        self.loss = loss_function()
        self.labels = labels

    def get_layers(self, has_param_only=True):
        for layer in filter(lambda x: x.param if has_param_only else True, self.layers):
            yield layer

    def predict(self, x, set_label=False):
        for layer in self.layers:
            x = layer.forward(x)

        if self.labels is not None and set_label:
            return [self.labels[idx] for idx in np.argmax(np.array(x), axis=1)]
        else:
            return x

    def get_accuracy(self, x, y):
        y_hat = self.predict(x, True)
        return sum(1 if _y_hat == _y else 0 for _y_hat, _y in zip(y_hat, y))/len(y_hat)

    def get_loss(self, x, y):
        z = self.predict(x)
        return self.loss.forward(z, y)

    def get_gradient(self, tr_x, tr_y):
        self.get_loss(tr_x, tr_y)
        dout = 1
        for layer in reversed(self.layers + [self.loss]):
            dout = layer.backward(dout)

