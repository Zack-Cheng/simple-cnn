import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.Utils import load_train_dataset

TEST_SET_IDX_FROM = 50000

cnn = None
with open('trained_model.pkl', 'rb') as f:
    cnn = pickle.load(f)

dataset, label = load_train_dataset()
dataset_size = dataset.shape[0]

for i in range(100):
    rand = random.randint(TEST_SET_IDX_FROM, dataset_size)

    img = dataset[rand:rand+1]
    actual = label[rand]
    predict = np.argmax(cnn.predict(img))

    smile = False
    if predict == actual:
        smile = True
        print('Dataset index {}: predict: \'{}\', actual: \'{}\'  {}'.format(
            rand,
            predict,
            actual,
            ':)' if smile else ':('
        ))

    plt.imshow(img[0, 0, :, :])
    plt.show(block=False)
    plt.pause(5)
    plt.close()
