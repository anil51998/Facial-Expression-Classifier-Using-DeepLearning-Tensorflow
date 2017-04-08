import pandas as pd 
import numpy as np 
import tensorflow as tf 
#for plot (incase needed)
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

df = pd.read_csv("~/Desktop/fer2013/fer2013.csv")
#df.head()
#np.unique(df["Usage"].values.ravel())  (ravel is used to flatten the data)
#print("Number of training data avail is: ", len(df[df."Usage"=='Training']))

#get the train data
train_data = df[df.Usage=='Training']
# convert all pixels values in train_data to string then split 
#and store the returned list to pixel_values
print("here")
pixels_values = train_data.pixels.str.split(" ").tolist()
print("done")
#convert these pixel_values to a dataFrame ( to apply pandas methods )
pixel_values = pd.DataFrame(pixels_values, dtype=int)
#put all the pixel values to images
images = pixel_values.values
images = images.astype(np.float)
#now images(a numpy array) contain the floating-point values (pixel_values)
images = images - images.mean(axis=1).reshape(-1,1)
images = np.multiply(images,100.0/255.0)
each_pixel_mean = images.mean(axis=0)
each_pixel_std = np.std(images, axis=0)

images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)

image_pixels = images.shape[1]
print('Flat pixel values is: ',image_pixels)
image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)

labels_flat = train_data["emotion"].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

VALIDATION_SIZE = 1709
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
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
y_ = tf.placeholder('float', shape=[None, labels_count])

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

conv_layer1 = tf.nn.relu(conv_layer(img, weight_conv1, "SAME")+ bias_conv1)
conv_pool1 = max_pool(conv_layer1)
conv_norm1 = tf.nn.lrn(conv_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#third:
weight_conv2 = weight_init([5, 5, 64, 128])
bias_conv2 = bias_init([128])

conv_layer2 = tf.nn.relu(conv_layer(conv_norm1, weight_conv2, "SAME")+ bias_conv2)
conv_norm2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
conv_pool2 = max_pool(conv_norm2)

#fully-connected layer: (convert this to conv layer by taking the filter size same as depth size)
# value 12 after two max-pooling (all these values are got after initial setting of hyper-parameters)
weight_full_layer1 = local_weight_init([12*12*128, 3072])
bias_full_layer1 = local_bias_init([3072])

full_shape1 = tf.reshape(conv_pool2,[-1, 12*12*128])
full1 = tf.nn.relu(tf.matmul(full_shape1, weight_full_layer1)+ bias_full_layer1)

# fully-connected layer 2:
weight_full_layer2 = local_weight_init([3072, 1536])
bias_full_layer2 = local_bias_init([1536])

full_shape2 = tf.reshape(full1, [-1, 3072])
full2 = tf.nn.relu(tf.matmul(full_shape2, weight_full_layer2)+bias_full_layer2)

# dropout to prevent overfitting

prob = tf.placeholder('float')
full2_drop = tf.nn.dropout(full2, prob)
# outputLayer:
w_output = weight_init([1536, labels_count])
b_output = bias_init([labels_count])

y = tf.nn.softmax(tf.matmul(full2_drop, w_output)+b_output)

LEARNING_RATE = 1e-4

#cost function:
cost = -tf.reduce_sum(y_*tf.log(y))

training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

#evaluation:
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#predict:
predict = tf.argmax(y, 1)

ITERATIONS = 3000
DROPOUT = 0.5
BATCH_SIZE = 50

epoch_count = 0
epoch_index = 0
num_examples = train_images.shape[0]

def batch(batch_size):
	global epoch_index
	global epoch_count
	global train_labels
	global train_images

	start = epoch_index
	epoch_index += batch_size
	if epoch_index > num_examples:
		epoch_count +=1
		perm = np.arange(num_examples)
		np.random.shuffle(perm)
		train_images = train_images[perm]
		train_labels = train_labels[perm]

		start = 0
		epoch_index = batch_size
		assert batch_size <= num_examples
	end = epoch_index
	return train_images[start:end], train_labels[start:end]

initial = tf.initialize_all_variables()
saver = tf.train.Saver()
session = tf.InteractiveSession()
session.run(initial)
#saver.restore(session, "model.ckpt")
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(ITERATIONS):
	xbatch, ybatch = batch(BATCH_SIZE)

	if i%display_step==0 or (i+1) == ITERATIONS:
		train_accuracy = accuracy.eval(feed_dict={x:xbatch, 
                                                  y_: ybatch, 
                                                  prob: 1.0})
		if(VALIDATION_SIZE):
			validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE], 
                                                            prob: 1.0})
			print("training_accuracy / validation_accuracy => ",train_accuracy, "/",validation_accuracy," for step", i)
			validation_accuracies.append(validation_accuracy)
		else:
			#print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
			print("comment2")
		train_accuracies.append(train_accuracy)
		x_range.append(i)
		if i%(display_step*10) == 0 and i and display_step<100:
			display_step *= 10
	session.run(training, feed_dict={x: xbatch, y_: ybatch, prob: DROPOUT})
save_path = saver.save(session, "model.ckpt")
