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

#l1
w1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
l1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides = [1, 1, 1, 1], padding = 'SAME'))
#l1_max_pool
l1_max_pool = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#[None, 16, 16, 32] 

#l2
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) 
l2 = tf.nn.relu(tf.nn.conv2d(l1_max_pool, w2, strides = [1, 1, 1, 1], padding = 'SAME'))
l2_max_pool = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#[None, 8, 8, 64]

#L3
w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
l3 = tf.nn.relu(tf.nn.conv2d(l2_max_pool, w3, strides = [1, 1, 1, 1], padding = 'SAME'))
#[None, 8, 8, 128] 

#L4
w4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
l4 = tf.nn.relu(tf.nn.conv2d(l3, w4, strides = [1, 1, 1, 1], padding = 'SAME'))
#[None, 8, 8, 256]

#l5
w5 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
l5 = tf.nn.relu(tf.nn.conv2d(l4, w5, strides=[1, 1, 1, 1], padding = 'SAME'))

#l6
w6 = tf.Variable(tf.random_normal([3, 3, 512, 600], stddev=0.01))
l6 = tf.nn.relu(tf.nn.conv2d(l5, w6, strides=[1, 1, 1, 1], padding = 'SAME'))
l6_max_pool = tf.nn.max_pool(l6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

dropout_rate = tf.placeholder(tf.float32)

#fc1
l6_flat = tf.reshape(l6_max_pool, [-1, 4*4*600])
w7 = tf.get_variable("w7", [4*4*600, 256], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.constant(0., shape=[256]))
l7 = tf.nn.relu(tf.matmul(l6_flat, w7) + b7)
l7_drop = tf.nn.dropout(l7, dropout_rate)
#[None, 256]

#fc2
w8 = tf.get_variable("w8", [256, 128], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.constant(0., shape=[128]))
l8 = tf.nn.relu(tf.matmul(l7_drop, w8) + b8)
l8_drop = tf.nn.dropout(l8, dropout_rate)
#[None, 384]

#fc3
w9 = tf.get_variable("w9", [128, 10], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.constant(0., shape=[10]))
#[None, 10]

hypothesis = tf.matmul(l8_drop, w9) + b9
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_epoch = 20 
train_size = 50000
batch_size  = 100

def next_batch(i, size, data, label):
    idx = i*size
    x_data = np.array(data[idx : idx + size, :])
    y_data = np.array(label[idx : idx + size, :])

    return x_data, y_data

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	
    for epoch in range(train_epoch):
        l = 0
        for step in range(train_size/batch_size):
            
            batch_xs, batch_ys = next_batch(step, batch_size, x_train, y_train_one_hot.eval())
            l, _ = sess.run([loss, optimizer], feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:0.5})

        print 'epoch : %02d / %02d ' %((epoch+1), train_epoch), 'cost : ', l, 'train accuracy : ', accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1})
        f_a = 0.
        for i in range(10):
            batch_xs, batch_ys = next_batch(i, 1000, x_test, y_test_one_hot.eval())
            a = accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})
            f_a += a

        print 'test accuracy : ', f_a/10
