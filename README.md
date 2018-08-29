# Simple CNN

Simple convolutional neural network implementation in Python

<br>

This implementation includes the following functions:

* Convolution / Affine / ReLU / Maxpooling / Softmax / Loss layers
* Numerical gradient computation
* Forward / backward propagation
* Cost computation
* Gradient checking

## Installation

```
pip3 install -r requirements.txt
```

## Usage

Run `python3 ./cnn.py` and train a convolutional neural network by [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

You can modify the parameters like *training epoch* and *batch size* in *cnn.py*.

Also a trained weights have been saved in this project using Python's pickle utility. 

Run `python3 ./demo.py` to see the trained model predicts the numbers of the MNIST dataset.

## Architecture

```
Conv    1: in: 28x28x1   window size: 5x5   stride: 1   out: 23x23x5
ReLu    1: in: 23x23x5   out: 23x23x5
MaxPool 1: in: 23x23x5   window size: 2x2   stride: 2   out: 12x12x5
Conv    2: in: 12x12x5   window size: 5x5   stride: 1   out: 8x8x2
ReLu    2: in: 8x8x2     out: 8x8x2
MaxPool 2: in: 8x8x2     window size: 2x2   stride: 1   out: 7x7x2
FC      1: in: 7x7x2     out: 98x1
Affine  1: in: 98        W: 98x50           out: 50
ReLu    3: in: 50        out: 50
Affine  2: in: 50        W: 50x10           out: 10
Softmax  : in: 10        out: 10
```

## Results

* Dev set size: 50000
* Test set size: 10000
* Training epoch: 10000
* Mini batch size: 100

<br>

* Dev set accuracy: 98.09%
* Test set accuracy: 97.63%

![accuracy](http://hzs.idv.tw/~zackcheng/github/simple_cnn/acc.png)

## Contact

If you encounter any problems while executing the program, feel free to report bugs or contact me by email at zaccy.cheng@gmail.com.
