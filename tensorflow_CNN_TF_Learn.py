#MNIST Dataset CNN
'''
REPEAT:
input > weight > hidden layer 1 (activation function) > weights > hiden l2
(activation function) > weights > output Layer

compare output to indented output > using cost or loss function (exp: cross entropy)
optimization function (optimizer) > minimize the cost (exp: Adamoptimizer, SGD, Adagrad)

backpropagation

feed forward + backprop = epoch (x times)
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
#from tensorflow.examples.tutorials.mnist import input_data

X,Y,test_x,test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

convnet = input_data(shape=[None,28,28,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation = 'relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation = 'relu')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet, 1024, activation='relu')
oonvnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

'''
#comment this out after fully TRAINED
model.fit({'input':X},{'targets':Y}, n_epoch=10, validation_set=({'input':test_x},{'targets':test_y}),
	snapshot_step=500, show_metric=True, run_id='mnist')
	
#saves weights of the model	
model.save('tflearncnn.model')
# Comment out
'''

#just load the last trained model and ready for test
model.load('tflearncnn.model')
#test on individual
print( model.predict( [test_x[2]] ) )
print('Original Label = ',test_y[2])

#Individual Test Image Show
IMG = test_x[2]
IMG = np.array(IMG, dtype='float')
pixels = IMG.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()




