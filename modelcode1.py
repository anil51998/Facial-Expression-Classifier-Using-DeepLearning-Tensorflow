import pandas as pd 
import numpy as np 
import tensorflow as tf 
#for plot (incase needed)
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

df = pd.read_csv("location of FER2013 dataset.csv")
#df.head()
#np.unique(df["Usage"].values.ravel())  (ravel is used to flatten the data)
#print("Number of training data avail is: ", len(df[df."Usage"=='Training']))

#get the train data
train_data = df[df."Usage"=='Training']
# convert all pixels values in train_data to string then split 
#and store the returned list to pixel_values
pixel_values = train_data.pixels.str.split("").tolist()
#convert these pixel_values to a dataFrame ( to apply pandas methods )
pixel_values = pd.DataFrame(pixel_values, dtype=int)
#put all the pixel values to images
images = pixel_values.values
images = images.astype(np.float)
#now images(a numpy array) contain the floating-point values (pixel_values)





# TENSORFLOW MODEL:

def weight_init(shape):
	#rather than assigning initial '0' weights assign a random weights (randomised based on normal distribution)
	init = tf.truncated_normal(shape, stddev=1e-4)
	return tf.Variable(init)

def bias_init(shape):
	init = tf.constant(0.1, shape=shape)
	return tf.Variable(init)

def local_weight_init(shape):
	init = tf.truncated_normal(shape, stddev=0.04)
	return tf.Variable(init)

def local_bias_init(shape):
	init = tf.constant(0.0, shape=shape)
	return tf.Variable(init)

def conv_layer(inp, weights, padd):
	# padding in the end if required.. 
	return tf.nn.conv2d(inp, weights, strides=[1,1,1,1], padding=padd)

def max_pool(inp):
	return tf.nn.max_pool(inp, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


#image_pixels yet to be calculated!! (Dataset unavail rn)	
#shape = height * width
x = tf.placeholder('float', shape=[None, image_pixels])
#labels unavail rn
y = tf.placeholder('float', shape=[None, labels_count])

# first convolution layer with 32 neurons (basically, feature mapping)
# second with 64 neurons
# third with 128 neurons
'''
model: hiddenl2 -> maxpool -> norm -> hidden3 -> norm -> maxpool -> FCl
'''
# first:
# 5,5-> (weight filter) are hyper-parameters
#weight_conv0 = weight_init([5, 5, 1, 32])
# a single bias for all 32 neurons
#bias_conv0 = bias_init([32])
'''
# batch, width, height, channel
#image_width and image_height to be defined
img = tf.reshape(x, [-1, image_width, image_height, 1])
conv_layer0 = tf.nn.relu(conv2d(img, weight_conv0, "SAME")+ bias_conv0)
'''
#second:
img = tf.reshape(x, [-1, image_width, image_height, 1])
weight_conv1 = weight_init([5, 5, 1, 64])
bias_conv1 = bias_init([64])

conv_layer1 = tf.nn.relu(conv2d(img, weight_conv1, "SAME")+ bias_conv1)
conv_pool1 = max_pool(conv_layer1)
conv_norm1 = tf.nn.lrn(conv_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#third:
weight_conv2 = weight_init([5, 5, 64, 128])
bias_conv2 = bias_init([128])

conv_layer2 = tf.nn.relu(conv2d(conv_norm1, weight_conv2, "SAME")+ bias_conv2)
conv_norm2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
conv_pool2 = max_pool(conv_norm2)

#fully-connected layer:
# value 12 after two max-pooling (all these values are got after initial setting of hyper-parameters)
weight_full_layer1 = local_weight_init([12*12*128, 3072])
bias_full_layer1 = local_bias_init([3072])
