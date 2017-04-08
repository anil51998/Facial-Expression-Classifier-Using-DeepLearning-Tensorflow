import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd 
import sklearn
from sklearn.model_selection import KFold

# clean the data and get these values
labels_count = 7
x = tf.placeholder("float", shape=[None, image_pixels])
y_ = tf.placeholder("float", shape=[None, labels_count])
# keyNote: image should be cropped to 224x224 pixels
image_width = 224
image_height = 224


batch_size = 128

	
# MODEL2:
def fc_weight_init(shape):
	init = tf.truncated_normal(shape, stddev=0.04)
	return tf.Variable(init)

def fc_bias_init(shape):
	init = tf.constant(0.0, shape=shape)
	return tf.Variable(init)

def  weight_init(shape):
	init = tf.truncated_normal(shape, stddev=1e-4)
	return tf.Variable(init)

def bias_init(shape):
	init = tf.constant(0.1, shape=shape)
	return tf.Variable(init)

def conv2d(input_image, weights, stride, padd):
	# trying:									can be CONSTANT, REFLECT & SYMMETRIC
	new_input_image = tf.pad(input_image, padd, "CONSTANT")
	return tf.nn.conv2d(new_input_image, weights, strides = stride, padding="SAME" )

def max_pool(input_image, kernel, stride, padd):
	new_input_image = tf.pad(input_image, padd, "CONSTANT")
	return tf.nn.max_pool(new_input_image, ksize=kernel, strides=stride, padding="SAME" )
	
def neural_network_model(x):
	
	img = tf.reshape([-1, image_width, image_height, 1])

	W_conv1 = weight_init([7, 7, 1, 64])
	b_conv1 = bias_init([64])
	# stride: 2, padding: 3
	conv1 = tf.nn.relu(conv2d(img, W_conv1, [1,2,2,1], [[3,3], [3,3]])+b_conv1)
	pool1 = max_pool(conv1, [1,3,3,1], [1,2,2,1], [[0,0],[0,0]])
	#lrn1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
	# using Alexnet finetuned values
	lrn1 = tf.nn.lrn(conv1, 2, bias=1.0, alpha=2e-05, beta=0.75, name="lrn1")

	# FeatEx1:
	W_conv2a = weight_init([1, 1, 64, 96])
	b_conv2a = bias_init([96])
	# stride: 1, padding: 0
	conv2a = tf.nn.relu(conv2d(lrn1, W_conv2a, [1,1,1,1], [[0,0],[0,0]])+b_conv2a)
	pool2a = max_pool(lrn1, [1,3,3,1], [1,1,1,1], [[1,1], [1,1]])
	W_conv2b = weight_init([3, 3, 96, 208])
	b_conv2b = bias_init([208])
	W_conv2c = weight_init([1, 1, 64, 64])
	b_conv2c = bias_init([64])
	# stride: 1, padding: 1
	conv2b = tf.nn.relu(conv2d(conv2a, W_conv2b, [1,1,1,1], [[1,1],[1,1]])+b_conv2b)
	conv2c = tf.nn.relu(conv2d(pool2a, W_conv2c, [1,1,1,1], [[0,0],[0,0]])+b_conv2c)
	concat2 = tf.concat([conv2b, conv2c], 0)
	pool2b = max_pool(concat2, [1,3,3,1], [1,2,2,1], [[0,0],[0,0]])


	# FeatEx2:
	W_conv3a = weight_init([1, 1, 64, 96])
	b_conv3a = bias_init([96])
	# stride: 1, padding: 0
	conv3a = tf.nn.relu(conv2d(pool2b, W_conv3a, [1,1,1,1], [[0,0],[0,0]])+b_conv3a)
	pool3a = max_pool(pool2b, [1,3,3,1], [1,1,1,1], [[1,1], [1,1]])
	W_conv3b = weight_init([3, 3, 96, 208])
	b_conv3b = bias_init([208])
	W_conv3c = weight_init([1, 1, 64, 64])
	b_conv3c = bias_init([64])
	# stride: 1, padding: 1
	conv3b = tf.nn.relu(conv2d(conv3a, W_conv3b, [1,1,1,1], [[1,1],[1,1]])+b_conv3b)
	conv3c = tf.nn.relu(conv2d(pool3a, W_conv3c, [1,1,1,1], [[0,0],[0,0]])+b_conv3c)
	concat3 = tf.concat([conv3b, conv3c], 0)
	pool3b = max_pool(concat3, [1,3,3,1], [1,2,2,1], [[0,0],[0,0]])

	#Fully-connected Layer:
	# have a check-up on 8192
	W_fully_connected = fc_weight_init([282*14*14, 8192])
	b_fully_connected = fc_bias_init([8192])
	fully_connected_shape = tf.reshape(pool3b, [-1, 282*14*14])
	fully_connected = tf.nn.relu(tf.matmul(fully_connected_shape, W_fully_connected)+b_fully_connected)
	# dropout:
	keep_prob = tf.placeholder('float')
	fully_connected_drop = tf.nn.dropout(fully_connected, keep_prob)
	# outputLayer:
	W_out = weight_init([8192, labels_count])
	b_out = bias_init([labels_count])
	y = tf.nn.softmax(tf.matmul(fully_connected_drop, W_out)+b_out)
	# 7 x 1 x 1 and not 11 x 1 x 1
	# 11 x 1 x 1 didn't make much sense!
	return y

def train_neural_network():
	# implement 10-fold cross-validation here ( using sklearn )
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	#default learn-rate: 0.001 or 1e-03
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	
