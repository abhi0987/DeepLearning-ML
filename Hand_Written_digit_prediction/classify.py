#import tensorflow and other libraries

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

seed=128
rng = np.random.RandomState(seed)

# Let's create the graph input of tensorflow by defining the 'Place Holders'
data_img_shape = 28*28  # 784 input units
digit_recognition = 10  # 10 classes : 0-9 digits or output units
hidden_num_units = 500  # hidden layer units

x = tf.placeholder(tf.float32,[None,data_img_shape])
y = tf.placeholder(tf.float32,[None,digit_recognition])

epochs = 5
batch_size = 128
learning_rate = 0.01
training_iteration = 50

# Let's define weights and biases of our model

# weights are the probablity that affects how data flow in the graph and 
# it will be updated continously during training
# so that our results will be closer to the right solution
weights = {
    'hidden' : tf.Variable(tf.random_normal([data_img_shape,hidden_num_units],seed=seed)),
    'output' : tf.Variable(tf.random_normal([hidden_num_units,digit_recognition],seed=seed))
}

# bias is to shift our regression line to better fit the data 
biases = {
     'hidden' : tf.Variable(tf.random_normal([hidden_num_units],seed=seed)),
     'output' : tf.Variable(tf.random_normal([digit_recognition],seed=seed))
}

# let's create our neural network computaional graph 

hidden_layer = tf.add(tf.matmul(x,weights['hidden']),biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.add(tf.matmul(hidden_layer,weights['output']),biases['output'])

# let's define our cost function 
# cost function minimize our erroe during training
# we will use cross entropy method to define the cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = y))


# let's set the optimizer i.e our backpropagation algorithim
# Here we use Adam, which is an efficient variant of Gradient Descent algorithm
# optimizer makes our model self improve through the training

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# We finish the architecture of our neural network
# nOw we will initialize all the variables

# Let's create an session and run our neural network in that session to train it


checkpoint_dir = "cps_py/"
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


ckpt = tf.train.get_checkpoint_state(checkpoint_dir)


if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("No checkpoint found ! Train the data")
    for iteration in range(1000):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(50):
            batch_x,batch_y =  mnist.train.next_batch(total_batch)  # create pre-processed batch
            _,c = sess.run([optimizer,cost],feed_dict = {x:batch_x , y:batch_y})   # feed the batch to optimizer

            avg_cost += c / total_batch          #find cost and reiterate to minimize

        print ("iteration :", (iteration+1), "cost =", "{:.5f}".format(avg_cost))
    print ("\nTraining complete!")

    #saving the session for later use
    saver.save(sess, checkpoint_dir+'model.ckpt')

pred_temp = tf.equal(tf.argmax(output_layer,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(pred_temp,'float'))


#print ("Validation Accuracy:", accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))   
print ("Validation Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})*100) 


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
    
    
    cv2.imwrite("data.png", gray)

    
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
    pred = prediction.eval({X: images})
    print("The prdicted number is : "+ str(pred))
    #print (sess.run(accuracy, feed_dict={X: image_, Y: image_.labels})*100) 


image = sys.argv[1]
imageprepare(x,y,image)        
