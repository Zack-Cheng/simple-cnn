import os
import requests
import gzip
import math
import pickle
import numpy as np
from urllib.request import urlopen
from tqdm import tqdm

DATASET_PATH = './dataset/'
TRAIN_DATASET_GZIP = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL_GZIP = 'train-labels-idx1-ubyte.gz'
TRAIN_DATASET_PATH = DATASET_PATH + TRAIN_DATASET_GZIP
TRAIN_LABEL_PATH = DATASET_PATH + TRAIN_LABEL_GZIP
DATASET_URL = 'http://yann.lecun.com/exdb/mnist/'
DATASET_PKL = DATASET_PATH + 'train-dataset.pkl'


def download_dataset(filename):
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    url = DATASET_URL + filename
    file_size = int(urlopen(url).info().get('Content-Length', -1))

    dst_path = DATASET_PATH + filename
    if os.path.exists(dst_path):
        first_byte = os.path.getsize(dst_path)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
                total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc='Download {}'.format(filename)
    )
    req = requests.get(url, headers=header, stream=True)
    with(open(dst_path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

def raw_to_data(raw_data):
    def _int(byte):
        return int.from_bytes(byte, 'big')

    data = []
    magic_num = _int(raw_data[:4])
    if magic_num == 2051:
        # image file
        img_num = _int(raw_data[4:8])
        row_num = _int(raw_data[8:12])
        col_num = _int(raw_data[12:16])

        for i in tqdm(range(img_num), desc='Loading datasets', leave=False):
            img = []
            for r in range(row_num):
                row = []
                for c in range(col_num):
                    row.append(raw_data[16 + i*row_num*col_num + r*col_num + c])
                img.append(row)
            data.append([img])

    elif magic_num == 2049:
        # label file
        img_item = _int(raw_data[4:8])
        for i in tqdm(range(img_item), desc='Loading labels', leave=False):
            data.append(raw_data[8+i])

    return data

def load_train_dataset():
    if os.path.exists(DATASET_PKL):
        with open(DATASET_PKL, 'rb') as f:
            out = pickle.load(f)
        return out

    if not os.path.exists(TRAIN_DATASET_PATH):
        download_dataset(TRAIN_DATASET_GZIP)

    if not os.path.exists(TRAIN_LABEL_PATH):
        download_dataset(TRAIN_LABEL_GZIP)

    with gzip.open(TRAIN_DATASET_PATH) as f:
        raw_dataset = f.read()

    with gzip.open(TRAIN_LABEL_PATH) as f:
        raw_label = f.read()

    dataset = np.array(raw_to_data(raw_dataset)) / 255.
    label = np.array(raw_to_data(raw_label))

    with open(DATASET_PKL, 'wb+') as f:
        pickle.dump((dataset, label), f)

    return dataset, label

def to_hot_vector(label):
        m = label.size
        t = np.zeros((10, m))
        t[label, np.arange(m)] = 1
        return t

def pad(x, pad_size):
    if x.ndim == 3:
        pad_width_args = ((0,0), (pad_size, pad_size), (pad_size, pad_size))
    elif x.ndim == 4:
        pad_width_args = ((0,0), (0,0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(
        x,
        pad_width=pad_width_args,
        mode='constant',
        constant_values=0
    )
def pad_each_element(x, pad_size):
    r, c = x.shape[-2:]
    r_idx = np.repeat(np.arange(1, r), pad_size)
    c_idx = np.repeat(np.arange(1, c), pad_size)
    padded_x = np.insert(np.insert(x, r_idx, 0, axis=1), c_idx, 0, axis=2)
    return padded_x

def output_num_of_conv(row_num, column_num, w_size, stride):
    cal = lambda x: (x - w_size) // stride + 1
    return cal(row_num), cal(column_num)

def conv_forward_flatten(x, w_size, pad_size=0, stride=1):
    padded_x = pad(x, pad_size)
    m, channels, rows, columns = padded_x.shape
    out_row, out_col = output_num_of_conv(rows, columns, w_size, stride)
    rst = np.zeros((m, out_row*out_col, channels*w_size**2))
    for r in range(0, rows - w_size + 1, stride):
        for c in range(0, columns - w_size + 1, stride):
            rst[:, r*out_col+c, :] = padded_x[:, :, r:r + w_size, c:c + w_size].reshape(m, -1)
    rst = rst.reshape(m*out_row*out_col, -1)
    return rst, out_row, out_col

def pool_forward_flatten(x, w_size, pad_size=0, stride=1):
    padded_x = pad(x, pad_size)
    m, channels, rows, columns = padded_x.shape
    out_row, out_col = output_num_of_conv(rows, columns, w_size, stride)
    rst = np.zeros((m, channels*out_row*out_col, w_size**2))
    for ch in range(0, channels):
        for r in range(0, rows - w_size + 1, stride):
            for c in range(0, columns - w_size + 1, stride):
                rst[:, ch*out_row*out_col+(r//stride)*out_col+c//stride, :] = padded_x[:, ch, r:r+w_size, c:c+w_size].reshape(m, -1)
    return rst, out_row, out_col

def ADAM_update(val, lr, m_t, n_t, it, grad):
    it += 1
    m_t = 0.9 * m_t + (1.-0.9) * grad
    n_t = 0.99 * n_t + (1.-0.99) * grad**2.
    m_t_correct = m_t / (1. - (0.9 ** it))
    n_t_correct = n_t / (1. - (0.99 ** it))
    val -= lr * (m_t_correct / (np.sqrt(n_t_correct) + 1e-7))
    return val, m_t, n_t, it

def numerical_gradient(f, x):
    h = 1e-5
    gradient = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        v1 = f(x)
        x[idx] = float(tmp_val) - h
        v2 = f(x)
        gradient[idx] = (v1 - v2) / (h*2)
        x[idx] = tmp_val

        it.iternext()
    return gradient
