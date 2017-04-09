import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd 
import sklearn
import os
import sys
import cv2
import random
from sklearn.model_selection import KFold

image_files = []
itype = "png"
image_dir = "cohn-kanade-images"
label_dir = "Emotion"
size = (224, 224)
labels_count = 7
image_width = 224
image_height = 224
final_image_array = []
labels = np.array([])
labels = labels.astype(np.uint8)
image_pixels = 0
def build_data():
	global image_files, itype, image_dir, label_dir, size, final_image_array, labels
	for outer_folder in os.listdir(image_dir):
		if os.path.isdir(image_dir + '/' + outer_folder):
			for inner_folder in os.listdir(image_dir + '/' + outer_folder):
				if os.path.isdir(image_dir + '/' + outer_folder + '/' + inner_folder):
					for input_file in os.listdir(image_dir + '/' + outer_folder + '/' + inner_folder):
						if input_file.split('.')[1] != itype:
							break
						label_file = label_dir+'/'+outer_folder+'/'+inner_folder+'/'+input_file[:-4] + '_emotion.txt'
						if os.path.isfile(label_file):
							read_file = open(label_file, 'r')
							label = int(float(read_file.readline()))
							for i in range(-1, -6, -1):
								image_file = sorted(os.listdir(image_dir + '/' + outer_folder + '/' + inner_folder))[i]
								if image_file.split('.')[1] == itype:
									image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+image_file, label))
								neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[0]
								if neutral_file.split('.')[1] != itype:
									neutral_file = sorted(os.listdir(image_dir+'/'+outer_folder+'/'+inner_folder))[1]
								image_files.append((image_dir+'/'+outer_folder+'/'+inner_folder+'/'+neutral_file, 0))

	for imagefile in image_files:
		imagefile = image_files[0]
		name = imagefile[0]
		label = imagefile[1]
		gray_img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
		resized_gray_img = cv2.resize(gray_img, size)
		resized_gray_img = resized_gray_img.astype(np.float)
		final_image_array.append(resized_gray_img.ravel())
		one_hot_arr = np.zeros((1, 8))
		one_hot_arr[0][label] = 1
		one_hot_arr = one_hot_arr.astype(np.uint8)
		labels = np.concatenate((one_hot_arr, labels))

def divide_data(test_percent=0.1):
    test_length = int(round(test_percent * len(final_image_array)))
    shuffled = final_image_array[:]
    random.shuffle(shuffled)
    training_data = shuffled[test_length:]
    testing_data = shuffled[:test_length]
    return training_data, testing_data

build_data()
image_pixels = final_image_array[0].shape[0]

x = tf.placeholder("float", shape=[None, image_pixels])
y_ = tf.placeholder("float", shape=[None, labels_count])


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
	return y

def train_neural_network():
	# implement 10-fold cross-validation here ( using sklearn )
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	#default learn-rate: 0.001 or 1e-03
	optimizer = tf.train.AdamOptimizer().minimize(cost)

