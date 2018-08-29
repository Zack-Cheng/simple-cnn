import numpy as np
from .Utils import conv_forward_flatten
from .Utils import pool_forward_flatten
from .Utils import pad
from .Utils import pad_each_element
from .Utils import ADAM_update

class Conv:
    def __init__(self, name, params):
        self.name = name
        self.pad = params[name]['pad']
        self.stride = params[name]['stride']
        self.w_w = params[name]['w_w']
        self.ic = params[name]['ic']
        self.oc = params[name]['oc']
        self.learn_rate = params['learn_rate']
        init_scale = params['init_scale']

        # D-1: output channel (oc)
        # D-2: input channel (ic)
        # D-3, D-4:  height / width of window (w_w)
        # shape of W: (D-1, D-2, D-3, D-4)
        self.w = init_scale * np.random.randn(self.oc, self.ic, self.w_w, self.w_w)
        self.m_t = np.zeros_like(self.w)
        self.n_t = np.zeros_like(self.w)
        self.update_iter = 0.

    def forward(self, x):
        self.x = x
        flatten_x, output_row_num, output_col_num = conv_forward_flatten(self.x,
                                                                         w_size=self.w_w,
                                                                         pad_size=self.pad,
                                                                         stride=self.stride)
        flatten_w = self.w.reshape(self.oc, -1).transpose()
        flatten_out = np.dot(flatten_x, flatten_w)
        out = flatten_out.reshape(x.shape[0],
                                  output_row_num,
                                  output_col_num,
                                  self.oc).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dw_flatten_x, *_ = conv_forward_flatten(self.x,
                                                w_size=self.w_w,
                                                pad_size=self.pad,
                                                stride=self.stride)
        dw_flatten_x = dw_flatten_x.T
        flatten_dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.oc)
        flatten_dw = np.dot(dw_flatten_x, flatten_dout)
        dw = flatten_dw.transpose().reshape(self.w.shape)

        pad_size = self.w_w - self.pad - 1
        padded_dout = pad_each_element(dout, self.stride - 1)
        padded_dout = pad(padded_dout, max(pad_size, 0))

        rotated_w = np.rot90(self.w, 2, axes=(3,2))
        flatten_padded_dout, out_row_num, out_col_num = conv_forward_flatten(padded_dout, self.w_w)
        flatten_rotated_w = rotated_w.reshape(
            self.w.shape[0],
            self.w.shape[1],
            -1
        ).transpose(0, 2, 1).reshape(self.w.shape[0]*self.w_w**2, -1)
        flatten_dx = np.dot(flatten_padded_dout, flatten_rotated_w)

        dx = flatten_dx.transpose().reshape(self.x.shape[1],
                                            self.x.shape[0],
                                            out_row_num,
                                            out_col_num).transpose(1, 0, 2, 3)
        if pad_size < 0:
            dx = dx[:, :, -pad_size:pad_size, -pad_size:pad_size]
        return dx, dw

    def update(self, dw):
        self.w, self.m_t, self.n_t, self.update_iter = ADAM_update(
            self.w,
            self.learn_rate,
            self.m_t,
            self.n_t,
            self.update_iter,
            dw
        )


class SoftmaxWithLoss:
    def softmax(self, x):
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def loss(self, y, t):
        return -np.sum(np.log(y) * t) / self.batch_size

    def forward(self, a, t):
        self.t = t
        self.batch_size = t.shape[1]
        y = self.softmax(a)
        self.y = y
        loss = self.loss(y, t)
        return loss

    def backward(self):
        return (self.y - self.t) / self.batch_size


class FullyConnected:
    def forward(self, x):
        self.x = x
        return x.reshape(x.shape[0], -1).transpose()

    def backward(self, dout):
        return dout.transpose().reshape(self.x.shape)


class Affine:
    def __init__(self, name, params):
        init_scale = params['init_scale']
        w_h = params[name]['w_h']
        w_w = params[name]['w_w']
        learn_rate = params['learn_rate']

        self.w = init_scale * np.random.randn(w_h, w_w)
        self.b = np.zeros((w_w, 1))
        self.m_t_w = np.zeros_like(self.w)
        self.m_t_b = np.zeros_like(self.b)
        self.n_t_w = np.zeros_like(self.w)
        self.n_t_b = np.zeros_like(self.b)
        self.update_iter = 0
        self.name = name
        self.x = None
        self.learn_rate = learn_rate

    def forward(self, x):
        x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        z = np.dot(self.w.transpose(), x) + self.b
        return z

    def backward(self, dout):
        dx = np.dot(self.w, dout)
        batch_size = self.x.shape[1]
        dw = np.dot(self.x, dout.transpose())
        db = np.sum(dout, axis=1).reshape(self.b.shape)
        return dx, dw, db

    def update(self, dw, db):
        self.w, self.m_t_w, self.n_t_w, _ = ADAM_update(
            self.w,
            self.learn_rate,
            self.m_t_w,
            self.n_t_w,
            self.update_iter,
            dw
        )

        self.b, self.m_t_b, self.n_t_b, self.update_iter = ADAM_update(
            self.b,
            self.learn_rate,
            self.m_t_b,
            self.n_t_b,
            self.update_iter,
            db
        )


class ReLU:
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout_copy = np.copy(dout)
        dout_copy[self.mask] = 0
        dx = dout_copy
        return dx


class MaxPool:
    def __init__(self, name, params):
        self.name = name
        self.w_w = params[name]['w_w']
        self.pad = params[name]['pad']
        self.stride = params[name]['stride']

    def forward(self, x):
        self.x = x
        flatten_x, output_row_num, output_col_num = pool_forward_flatten(self.x,
                                                                         w_size=self.w_w,
                                                                         pad_size=self.pad,
                                                                         stride=self.stride)
        flatten_rst = np.max(flatten_x, axis=2)
        self.mask = np.argmax(flatten_x, axis=2)
        rst = flatten_rst.reshape(x.shape[0], x.shape[1], output_row_num, output_col_num)
        return rst

    def backward(self, dout):
        dmax_flatten = np.zeros((self.w_w**2, dout.size))
        m = dout.shape[0]
        dmax_flatten[self.mask, np.arange(dout.size).reshape(m, -1)] = dout.reshape(m, -1)[:]
        dmax = dmax_flatten.transpose().reshape(dout.shape[0],
                                                dout.shape[1],
                                                dout.shape[2],
                                                dout.shape[3],
                                                self.w_w,
                                                self.w_w)
        dx = np.zeros(self.x.shape)
        dx = pad(dx, self.pad)
        for r in range(dout.shape[2]):
            for c in range(dout.shape[3]):
                dx[:, :, r*self.stride:r*self.stride + self.w_w, c*self.stride:c*self.stride + self.w_w] += dmax[:, :, r, c]
        if self.pad != 0:
            dx = dx[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dx

