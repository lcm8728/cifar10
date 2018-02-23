import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

#load cifar10 datasets
(x_train, y_train), (x_test, y_test) = np.array(tf.keras.datasets.cifar10.load_data())
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10, axis = 1))
y_test_one_hot  = tf.squeeze(tf.one_hot(y_test, 10, axis = 1))
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)

def filter(h, w, c, n):
    return tf.Variable(tf.random_normal([h, w, c, n], stddev = 0.01))

def convolution2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

class inception():
    def __init__(self, x, c, vol):
        self.wa = filter(1, 1, c, vol/4)
        self.wb = filter(1, 1, c, vol/8)
        self.wc = filter(3, 3, vol/8, vol/2)
        self.wd = filter(1, 1, c, vol/4)
        
        self.conva = convolution2d(x, self.wa)
        self.convb = convolution2d(x, self.wb)
        self.convc = convolution2d(self.convb, self.wc)
        self.convd = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
        self.conve = convolution2d(self.convd, self.wd)
        
        self.conv = tf.concat([self.conva, self.convc, self.conve], 3)
       
        self.gamma = tf.Variable(tf.constant(1.0, shape=[vol]))
        self.beta = tf.Variable(tf.constant(0.0, shape=[vol]))
        self.mean, self.variance = tf.nn.moments(self.conv, [0, 1, 2])
        self.conv = tf.nn.batch_normalization(self.conv, self.mean, self.variance, self.beta, self.gamma, 0.01)
        
        self.conv = tf.nn.relu(self.conv)
    
    def maxpool(self):
        self.conv = tf.nn.max_pool(self.conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#MODEL
w1a = filter(1, 1, 3, 16)
w1b = filter(1, 1, 3, 1)
w1c = filter(3, 3, 1, 32)
w1d = filter(1, 1, 3, 16)

conv1a = convolution2d(X, w1a)
conv1b = convolution2d(X, w1b)
conv1c = convolution2d(conv1b, w1c)
conv1d = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
conv1e = convolution2d(conv1d, w1d)

conv1 = tf.concat([conv1a, conv1c, conv1e], 3)
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = inception(conv1, 64, 128)
conv2_1 = inception(conv2.conv, 128, 128)
conv2_2 = inception(conv2_1.conv, 128, 128)
conv2_2.maxpool()
conv3 = inception(conv2_2.conv, 128, 192)
conv3_1 = inception(conv3.conv, 192, 192)
conv3_1.maxpool()
conv4 = inception(conv3_1.conv, 192, 256)
conv5 = inception(conv4.conv, 256, 256)
conv6 = inception(conv5.conv, 256, 256)
conv6.maxpool()

#fc1
conv6_flat  = tf.reshape(conv6.conv, [-1, 2*2*256])
w7 = tf.Variable(tf.random_normal([2*2*256, 256], stddev = 0.01))
b7 = tf.Variable(tf.random_normal([256], stddev = 0.01))
fc1 = tf.nn.relu(tf.matmul(conv6_flat, w7) + b7)
fc1 = tf.nn.dropout(fc1, dropout_rate)

#fc2
w8 = tf.Variable(tf.random_normal([256, 256], stddev = 0.01))
b8 = tf.Variable(tf.random_normal([256], stddev = 0.01))
fc2 = tf.nn.relu(tf.matmul(fc1, w8) + b8)
fc2 = tf.nn.dropout(fc2, dropout_rate)

#fc3
w9 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01))
b9 = tf.Variable(tf.zeros([10]))

hypothesis = tf.matmul(fc2, w9) + b9
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_epoch = 300 
train_size = 50000
batch_size  = 100

def next_batch(i, size, data, label):
    idx = i*size
    x_data = np.array(data[idx : idx + size, :])
    y_data = np.array(label[idx : idx + size, :])

    return x_data, y_data

x_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=45)

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	
    for epoch in range(train_epoch):
        l = 0
        for step in range(train_size/batch_size):
            
            batch_xs, batch_ys = next_batch(step, batch_size, x_train, y_train_one_hot.eval())
            batch_xs_gen = x_data_gen.flow(batch_xs)
            l, _ = sess.run([loss, optimizer], feed_dict={X:batch_xs_gen, Y:batch_ys, dropout_rate:0.7})

        print 'epoch : %02d / %02d ' %((epoch+1), train_epoch), 'cost : ', l, 'train accuracy : ', accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1})
        f_a = 0.
        for i in range(10):
            batch_xs, batch_ys = next_batch(i, 1000, x_test, y_test_one_hot.eval())
            a = accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})
            f_a += a

        print 'test accuracy : ', f_a/10
