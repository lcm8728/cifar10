import tensorflow as tf
import numpy as np
import random
import time
from progress.bar import Bar

#load cifar10 datasets
(x_train, y_train), (x_test, y_test) = np.asarray(tf.keras.datasets.cifar10.load_data())
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10, axis = 1))
y_test_one_hot  = tf.squeeze(tf.one_hot(y_test, 10, axis = 1))
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

def filter(h, w, c, n):
    return tf.Variable(tf.random_normal([h, w, c, n], stddev = 0.01))

def convolution2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

class inception():
    def __init__(self, x, c, vol):
        self.wa = filter(1, 1, c, int(vol/4))
        self.wb = filter(1, 1, c, int(vol/8))
        self.wc = filter(3, 3, int(vol/8), int(vol/2))
        self.wd = filter(1, 1, c, int(vol/4))

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
w1 =filter(3, 3, 3, 64)
conv1 = convolution2d(X, w1)
conv2 = inception(conv1, 64, 128)
conv2.maxpool()
conv3 = inception(conv2.conv, 128, 192)
conv3.maxpool()
conv4 = inception(conv3.conv, 192, 192)
conv5 = inception(conv4.conv, 192, 256)
conv5.maxpool()
conv6 = inception(conv5.conv, 256, 256)
conv6.maxpool()

#fc1
conv6_flat  = tf.reshape(conv6.conv, [-1, 2*2*256])
w7 = tf.Variable(tf.random_normal([2*2*256, 256], stddev = 0.01))
b7 = tf.Variable(tf.random_normal([256], stddev = 0.01))
fc1 = tf.nn.relu(tf.matmul(conv6_flat, w7) + b7)

#fc2
w8 = tf.Variable(tf.random_normal([256, 256], stddev = 0.01))
b8 = tf.Variable(tf.random_normal([256], stddev = 0.01))
fc2 = tf.nn.relu(tf.matmul(fc1, w8) + b8)

#fc3
w9 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01))
b9 = tf.Variable(tf.zeros([10]))

hypothesis = tf.matmul(fc1, w9) + b9
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_epoch = 300 
train_size = 50000
test_size = 10000
batch_size  = 500

def next_batch(num, batch_size, x, y):
	x_data = x[num*batch_size:num*batch_size+batch_size, :]
	y_data = y[num*batch_size:num*batch_size+batch_size, :]

	return x_data, y_data

#train
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    
	gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	aug_data = gen.flow(x_train, y_train, batch_size = batch_size, shuffle=False)
	aug_data.y = tf.squeeze(tf.one_hot(aug_data.y, 10, axis = 1)).eval()
    
	for epoch in range(train_epoch):
		l = 0
		start_time = time.time()
		bar = Bar("Processing", max = train_size/batch_size)
		for step in range(int(train_size/batch_size)):
			batch_train_x = aug_data[step][0]
			batch_train_y = aug_data[step][1]
			l, _ = sess.run([loss, optimizer], feed_dict={X:batch_train_x, Y:batch_train_y})
			bar.next()
		bar.finish()
		print("epoch : %02d / %02d"%((epoch+1), train_epoch), "cost : ", l, "train accuracy : ", accuracy.eval(feed_dict={X:batch_train_x, Y:batch_train_y}), "time : %.2f"%(time.time()-start_time))
        
		test_accuracy = 0.
		num_iter = int(test_size/batch_size)
		for step in range(num_iter):
			batch_test_x, batch_test_y = next_batch(step, batch_size, x_test, y_test_one_hot.eval())
			a = accuracy.eval(feed_dict={X:batch_test_x, Y:batch_test_y})
			test_accuracy += a

		print("test accuracy : %.2f"%(a/num_iter))
