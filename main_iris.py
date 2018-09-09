# -*- coding: utf-8 -*-

import os
import datetime as dt
from utils_prev import *

# setting
batch_size = 30
n_epoch = 2000
training_x, training_y = [], []
test_x, test_y = None, None

# Load data
dat_path = os.path.abspath(__file__ + '/../../') + '\data\iris.csv'
with open(dat_path, 'r') as f:
    tmp_dat = f.readlines()
    print(tmp_dat.pop(0).split(','))

    for row in tmp_dat:
        row = row.split(',')
        training_y.append(row.pop().replace('\n', ''))  # label
        training_x.append(list(map(float, row)))  # x

training_x = np.array(training_x)
training_y, labels = one_to_hot(training_y)
print(training_x.shape)
print(labels)

# Make Neural Network
layers = list()
layers.append(Input(4))  # input layer
layers.append(FullyConnected(4, 3))  # fully connected with softmax activation
layers.append(Softmax())

nn = NeuralNetwork(layers, CrossEntropy, labels)
sgd = SGD(0.001)

for epoch_idx in range(n_epoch):

    if epoch_idx % 100 == 0:
        now_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        accuracy = nn.get_accuracy(training_x, [labels[np.argmax(y, axis=0)] for y in training_y])
        print('epoch: {epoch}, {now_time}, {accuracy:.4f}'.format(epoch=epoch_idx,
                                                                  now_time=now_time,
                                                                  accuracy=accuracy))
    for tr_x, tr_y in next_batch(batch_size, x=training_x, y=training_y):
        sgd.update(nn, tr_x, tr_y)
