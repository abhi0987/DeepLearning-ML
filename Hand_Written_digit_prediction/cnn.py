import matplotlib.pyplot as plt
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import accuracy_score
import tensorflow as tf
from scipy.misc import imread
from PIL import Image, ImageFilter
import cv2

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
seed = 128
# Let's create the graph input of tensorflow by defining the 'Place Holders'

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

batch_size = 128

'''
Weight Initialization
'''


def Wight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,seed=seed)
    return tf.Variable(initial)


def Bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
Convolutional and Pooling

Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
Our pooling is plain old max pooling over 2x2 blocks.
To keep our code cleaner, let's also abstract those operations into functions.'''


def Convo_2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def Maxpool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
We can now implement our first layer. It will consist of convolution, followed by max pooling.
The convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels.
We will also have a bias vector with a component for each output channel.'''

w_conv1 = Wight_var([5, 5, 1, 32])
b_conv1 = Bias_var([32])

'''reshape x to a 4d tensor with 2nd and third dimension to be width & height and final variable be the no of. color channels'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

# then convolve x_image with  weight tensor , add bias and apply Relu functio then Maxpooling
# Max pooling reduce the image size to 14*14

h_conv1 = tf.nn.relu(Convo_2D(x_image, w_conv1) + b_conv1)
h_pool1 = Maxpool_2x2(h_conv1)

'''Second layer :
to build a deep network we have to create several hidden layers of the above type
the 2nd layer have 64 features for each 5*5 patch'''

w_conv2 = Wight_var([5, 5, 32, 64])
b_conv2 = Bias_var([64])

h_conv2 = tf.nn.relu(Convo_2D(h_conv1, w_conv2) + b_conv2)
h_pool2 = Maxpool_2x2(h_conv2)

'''
Here the conversion of 2/3d images to 1D vectors'''
h_pool2_shape = h_pool2.get_shape()

# Don't hard-code the 1D vector dim. Rather, (1) multiply image's height,
# width and depth to get it.
h_pool2_dim = h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3]

'''
Fully connected layer :

Now image size reduced to 7*7 , next we add a fully connected layer
with 1024 neurons to allow processing on the entire image , we reshape the tensor from pooling layer into a batch of vectors
multiply by a weight matrix, add a bias and apply RELU
 '''

#  Use the computed 1D dimension to set the FC1 weight matrix dimensions.
w_fc1 = Wight_var(tf.stack([h_pool2_dim, 1024]))
b_fc1 = Bias_var([1024])

# Use the same 1D dimension to correctly reshape the batch matrix.
h_pool2_flat = tf.reshape(h_pool2, tf.stack([-1, h_pool2_dim]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

'''Drop out layer :
to reduce overfitting we use drop out layer before readout layer'''

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''Readout Layer :
finally we add readout layer'''

w_fc2 = Wight_var([1024, 10])
b_fc2 = Bias_var([10])

output_layer = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

# cost function to minimize errors during traing
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

# optimizer makes our model self improve through the training
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

# create to a checkpoint directory to save the trained model
checkpoint_dir = "checkpoint_dir/"
saver = tf.train.Saver()

pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_temp, 'float'))
# Let's create an session and run our neural network in that session to train it
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("model restored")
else:
    print("No checkpoint found ! Train the data")

    for iteration in range(500):
        avg_cost = 0;
        
        for i in range(20):
            
            batch_x, batch_y = mnist.train.next_batch(32)  # create pre-processed batch
            _, c = sess.run([optimizer, cost],
                            feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})  # feed the batch to optimizer
            avg_cost += c / 32  # find cost and reiterate to minimize
            print("iteration :", (iteration + 1), "cost =", "{:.5f}".format(avg_cost))
    print("\nTraining complete!\n")

    # saving the session for later use
    saver.save(sess, checkpoint_dir + 'model.ckpt')

print("Now printing accuarcy ::: ", sess.run(accuracy, feed_dict={x: np.split(mnist.test.images,5)[0], y: np.split(mnist.test.labels,5)[0], keep_prob: 1.0}) * 100)

import math

# get the best shift value for shifting
def getBestShift(img):
    cx,cy = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftX = np.round(cols/2.0-cx).astype(int)
    shiftY = np.round(rows/2.0-cy).astype(int)
    return shiftX,shiftY

# shift the img to the center
def shift(img,shiftx,shifty):
    rows,cols = img.shape
    M = np.float32([[1,0,shiftx],[0,1,shifty]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def imageprepare(X,Y,data):
    #create an array to store the eight images
    images = np.zeros((1,784))

    #array to store correct values
    correct_vals = np.zeros((1,10))
    
    gray = cv2.imread(data,0)
    
    # resize the images and invert it (black background) 
    gray = cv2.resize(255-gray,(28,28))
    
    #Okay it's quite obvious that the images doesn't 
    #look like the trained ones. These are white digits on a gray background and not on a black one.
    
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
     
    """All images are size normalized to fit in a
    20x20 pixel box and there are centered in a 28x28 image 
    using the center of mass. These are important information for our preprocessing."""
    
    """First we want to fit the images into this 20x20 pixel box.
    Therefore we need to remove every row and column at the sides of the image which are completely black"""
    
    while np.sum(gray[0]) == 0:
          gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
          gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
          gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
          gray = np.delete(gray,-1,1)

    rows,cols = gray.shape
    
    """Now we resize our outer box to fit it into a 20x20 box. Let's calculate the resize factor:"""
    
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))
    
    """But at the end we need a 28x28 pixel image so we add the missing black
    rows and columns using the np.lib.pad function which adds 0s to the sides."""
        
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    
    
    """ Here comes the shift operation """
    
    #shiftx,shifty = getBestShift(gray)
    #shifted = shift(gray,shiftx,shifty)
    #gray = shifted
    
    
    cv2.imwrite("edited/"+"data.png", gray)

    
    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    
    """The next step is to shift the inner box so that it is centered using the center of mass."""
    
    
    
    
    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    
    
    images[0] = flatten
    #correct_val = np.zeros((10))
    #correct_val[0] = 1
    #correct_vals[0] = correct_val
    

    prediction = tf.argmax(output_layer,1)
    
    
    """
    we want to run the prediction and the accuracy function
    using our generated arrays (images and correct_vals)
    """
    pred = prediction.eval({X: images,keep_prob:1.0})
    print("The prdicted number is : "+ str(pred))
    print (sess.run(accuracy, feed_dict={X: image_, Y: image_.labels , keep_prob:1.0})*100) 



'''Enter the path of image with python file'''
image = sys.argv[1]
imageprepare(x,y,image)  