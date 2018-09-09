# -*- coding: utf-8 -*-

import numpy as np


def im2col(img, fh, fw, s=1, p=0):
    """ Form matrix from image so that make convolution operation to matrix product.
    :param img: (n, h, w, c) shape of numpy array,
    :param fh: int, height of filter.
    :param fw: int, width of filter.
    :param s: int, stride of filter. Defaults to 1.
    :param p: int, zero-padding for image. Defaults to 0.
    :return: numpy array, matrix form of image.
    """

    n, h, w, c = img.shape  # n:number of images, h:height, w:width, c:channel
    oh = (h + 2*p - fh)//s + 1  # convolution output height
    ow = (w + 2*p - fw)//s + 1  # convolution output width

    assert oh*fh >= h+p and ow*fw >= w+p  # prevent loss of information

    out = np.zeros((n*oh*ow, fh*fw*c))  # im2col output
    k = 0  # row index of out
    img = np.pad(img, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')  # zero padding

    for j in range(ow):
            for i in range(oh):
                tmp_img = img[:, i*s:(i*s+fh), j*s:(j*s+fw), :].transpose(0, 3, 2, 1).reshape((n, -1))
                out[np.arange(n)*ow*oh+k, :] = tmp_img
                k += 1

    return out


def col2im(col, h, w, fh, fw, s=1, p=0):
    """ Form image from im2col matrix.
    :param col: (n*oh*ow, fh*fw*c) matrix
    :param h: int, height of image.
    :param w: int, width of image.
    :param fh: int, height of filter.
    :param fw: int, width of filter.
    :param s: int, stride of filter. Defaults to 1.
    :param p: int, zero-padding for image. Defaults to 0.
    :return: (n, h, w, c) shape of numpy array,
    """

    h_p, w_p = h + 2*p, w + 2*p
    oh = (h + 2*p - fh)//s + 1  # convolution output height
    ow = (w + 2*p - fw)//s + 1  # convolution output width

    n_oh_ow, fh_fw_c = col.shape
    n, c = n_oh_ow // (oh*ow), fh_fw_c // (fh*fw)

    col = col.reshape((n, oh, ow, c, fh, fw)).transpose((0, 1, 2, 5, 4, 3))
    out = np.zeros((n, h_p, w_p, c))

    for i in range(ow):
        for j in range(oh):
            tmp_img = col[:, j, i, :, :]
            out[:, i*s:(i*s+fh), j*s:(j*s+fw), :] = tmp_img

    return out if p ==0 else out[:, p:-p, p:-p, :]


class Layer:

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class Input(Layer):

    def forward(self, x):
        pass


if __name__ == '__main__':

    # im2col, col2im test
    print('im2col, col2im test')
    n_, h_, w_, c_ = 20, 10, 10, 3
    img_ = np.arange(n_*h_*w_*c_).reshape((n_, h_, w_, c_))
    fh_, fw_ = 2, 2
    p_, s_ = 1, 1

    im2col_out = im2col(img_, fh_, fw_, s_, p_)
    col2im_out = col2im(im2col_out, h_, w_, fh_, fw_, s_, p_)

    print(np.sum(np.abs(img_-col2im_out)) == 0.0)


