from lib.Layers import Conv
from lib.Layers import MaxPool
from lib.Layers import SoftmaxWithLoss
from lib.Layers import FullyConnected
from lib.Layers import Affine
from lib.Layers import ReLU

class CNN:
    def __init__(self):
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
            'conv2': {
                'pad': 0,
                'stride': 1,
                'w_w': 5,
                'ic': 5,
                'oc': 2
            },
            'pool1': {
                'w_w': 2,
                'pad': 0,
                'stride': 2
            },
            'pool2': {
                'w_w': 2,
                'pad': 0,
                'stride': 1
            },
            'affine1': {
                'w_h': 98,
                'w_w': 50
            },
            'affine2': {
                'w_h': 50,
                'w_w': 10
            }
        }

        conv1 = Conv(name='conv1', params=params)
        conv2 = Conv(name='conv2', params=params)
        relu1 = ReLU()
        relu2 = ReLU()
        relu3 = ReLU()
        pool1 = MaxPool('pool1', params=params)
        pool2 = MaxPool('pool2', params=params)
        fc = FullyConnected()
        affine1 = Affine(name='affine1', params=params)
        affine2 = Affine(name='affine2', params=params)
        self.softmax_with_loss = SoftmaxWithLoss()

        self.layers = [
            conv1, relu1, pool1,
            conv2, relu2, pool2,
            fc,
            affine1, relu3,
            affine2
        ]

    def predict(self, x):
        out = self.layers[0].forward(x)
        for layer in self.layers[1:]:
            out = layer.forward(out)
        return out

    def forward_prop(self, x, t):
        out = self.predict(x)
        loss = self.softmax_with_loss.forward(out, t)
        return loss

    def backward_prop(self):
        dout = self.softmax_with_loss.backward()
        for layer in reversed(self.layers):
            if isinstance(layer, Affine):
                dout, dw, db = layer.backward(dout)
                layer.update(dw, db)
            elif isinstance(layer, Conv):
                dout, dw = layer.backward(dout)
                layer.update(dw)
            else:
                dout = layer.backward(dout)

