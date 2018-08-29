import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from lib.Utils import load_train_dataset
from lib.Utils import to_hot_vector
from lib.Network import CNN

## PARAMETERS ##
BATCH_SIZE = 100
ITER_TRAIN_SIZE = 10000
TRAIN_EPOCH = 100
TRAIN_SET_SIZE = 50000 # total = 60000, test set size = total - train set size
################

dataset, label = load_train_dataset()
cnn = CNN()

dataset_size = dataset.shape[0]
train_accs = []
test_accs = []
for epoch in range(TRAIN_EPOCH):
    random_idx = np.random.choice(TRAIN_SET_SIZE, ITER_TRAIN_SIZE)
    iter_dataset = dataset[random_idx]
    iter_label = label[random_idx]
    for i in trange(0, ITER_TRAIN_SIZE, BATCH_SIZE,
                    desc='Training epoch {}'.format(epoch+1),
                    leave=False):
        x = iter_dataset[i:i+BATCH_SIZE]
        t = to_hot_vector(iter_label[i:i+BATCH_SIZE])
        loss = cnn.forward_prop(x, t)
        cnn.backward_prop()

    def _acc(from_data_idx, to_data_idx, desc=None):
        total_match_num = 0.
        total_size = to_data_idx - from_data_idx
        for i in trange(from_data_idx, to_data_idx, BATCH_SIZE,
                        desc='Evaluating epoch {} {} accuracy'.format(
                            epoch+1,
                            desc or ''
                        ),
                        leave=False):
            x = dataset[i:i+BATCH_SIZE]
            actual = label[i:i+BATCH_SIZE]
            predict = np.argmax(cnn.predict(x), axis=0)
            match_num = len(np.where(actual == predict)[0])
            total_match_num += match_num
        return total_match_num / total_size

    train_accs.append(_acc(0, TRAIN_SET_SIZE, 'train'))
    test_accs.append(_acc(TRAIN_SET_SIZE, dataset_size, 'test'))

    print('epoch: {}, train acc: {}, test acc: {}'.format(
        epoch + 1,
        train_accs[-1],
        test_accs[-1]
    ))

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(cnn, f)

plt.plot(train_accs)
plt.plot(test_accs)
plt.legend(['Training set acc.', 'Test set acc.'])
plt.show()

