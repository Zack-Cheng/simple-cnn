import numpy as np
from lib.Utils import numerical_gradient
from lib.Utils import to_hot_vector
from lib.Layers import Conv
from lib.Layers import MaxPool
from lib.Layers import SoftmaxWithLoss
from lib.Layers import FullyConnected
from lib.Layers import Affine
from lib.Layers import ReLU

params = {
    'init_scale': 0.01,
    'learn_rate': 0.001,
    'conv1': {
        'pad': 0,
        'stride': 1,
        'w_w': 5,
        'ic': 1,
        'oc': 5
    },
    'pool1': {
        'w_w': 2,
        'pad': 0,
        'stride': 2
    },
    'affine1': {
        'w_h': 45,
        'w_w': 50
    },
    'affine2': {
        'w_h': 50,
        'w_w': 10
    }
}

conv1 = Conv(name='conv1', params=params)
relu1 = ReLU()
relu2 = ReLU()
pool1 = MaxPool('pool1', params=params)
fc = FullyConnected()
affine1 = Affine(name='affine1', params=params)
affine2 = Affine(name='affine2', params=params)
softmax_with_loss = SoftmaxWithLoss()

layers = [conv1, relu1, pool1,
          fc,
          affine1, relu2,
          affine2]

def loss(x, t):
    out = layers[0].forward(x)
    for layer in layers[1:]:
        out = layer.forward(out)
    loss = softmax_with_loss.forward(out, t)
    return loss

def gradient(x, t):
    grad = {
        'conv1': {
            'dw': 0
        },
        'conv2': {
            'dw': 0
        },
        'affine1': {
            'dw': 0,
            'db': 0
        },
        'affine2': {
            'dw': 0,
            'db': 0
        }
    }

    loss(x, t)
    dout = softmax_with_loss.backward()
    for layer in reversed(layers):
        if isinstance(layer, Affine):
            dout, dw, db = layer.backward(dout)
            grad[layer.name]['dw'] = dw
            grad[layer.name]['db'] = db
        elif isinstance(layer, Conv):
            dout, dw = layer.backward(dout)
            grad[layer.name]['dw'] = dw
        else:
            dout = layer.backward(dout)
    return grad

def num_gradient(x, t):
    grad = {
        'conv1': {
            'dw': 0
        },
        'affine1': {
            'dw': 0,
            'db': 0
        },
        'affine2': {
            'dw': 0,
            'db': 0
        }
    }

    def _loss_g(w):
        return loss(x, t)

    grad['conv1']['dw'] = numerical_gradient(_loss_g, layers[0].w)
    grad['affine1']['dw'] = numerical_gradient(_loss_g, layers[4].w)
    grad['affine1']['db'] = numerical_gradient(_loss_g, layers[4].b)
    grad['affine2']['dw'] = numerical_gradient(_loss_g, layers[6].w)
    grad['affine2']['db'] = numerical_gradient(_loss_g, layers[6].b)
    return grad


if __name__ == '__main__':
    x = np.random.randn(1, 1, 10, 10)
    t = to_hot_vector(np.array([1]))
    grad = gradient(x, t)
    num_grad = num_gradient(x, t)

    def _diff(name, param):
        return np.abs(grad[name][param] - num_grad[name][param]).mean()

    print('conv1 dw diff: {}'.format(_diff('conv1', 'dw')))
    print('affine1 dw diff: {}'.format(_diff('affine1', 'dw')))
    print('affine1 db diff: {}'.format(_diff('affine1', 'db')))
    print('affine2 dw diff: {}'.format(_diff('affine2', 'dw')))
    print('affine2 db diff: {}'.format(_diff('affine2', 'db')))

