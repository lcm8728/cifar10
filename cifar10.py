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
#X.shape=[None, 32, 32, 3]
#Y.shape=[None, 10]

#L1
w1 = tf.Variable(tf.random_normal([8, 8, 3, 32], stddev=0.01))
l1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides = [1, 1, 1, 1], padding = 'SAME'))
l1_max_pool = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#L1.shape=[None, 16, 16, 32] 

#L2
w2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)) 
l2 = tf.nn.relu(tf.nn.conv2d(l1_max_pool, w2, strides = [1, 1, 1, 1], padding = 'SAME'))
l2_max_pool = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#L2.shape=[None, 8, 8, 64]

#L3
w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
l3 = tf.nn.relu(tf.nn.conv2d(l2_max_pool, w3, strides = [1, 1, 1, 1], padding = 'SAME'))
#L3.shape=[None, 8, 8, 128] 

#L4
w4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
l4 = tf.nn.relu(tf.nn.conv2d(l3, w4, strides = [1, 1, 1, 1], padding = 'SAME'))
#L4.shape=[None, 8, 8, 128]

dropout_rate = tf.placeholder(tf.float32)

#FC1
l4_flat = tf.reshape(l4, [-1, 8*8*128])
w5 = tf.get_variable("w5", [8*8*128, 128], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.constant(0., shape=[128]))
l5 = tf.nn.relu(tf.matmul(l4_flat, w5) + b5)
l5_drop = tf.nn.dropout(l5, dropout_rate)

#FC2
w6 = tf.get_variable("w6", [128, 384], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.constant(0., shape=[384]))
l6 = tf.nn.relu(tf.matmul(l5_drop,w6) + b6)
l6_drop = tf.nn.dropout(l6, dropout_rate)

#FC3
w7 = tf.get_variable("w7", [384, 10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.constant(0., shape=[10]))

hypothesis = tf.matmul(l6_drop, w7) + b7
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_epoch = 15 
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
            h, l, _ = sess.run([accuracy, loss, optimizer], feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:0.7})

        print 'epoch : %04d / ' %(epoch+1), train_epoch, ' cost : ', l, ' accuracy : ', accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1})
        f_a = 0.
        for i in range(10):
            batch_xs, batch_ys = next_batch(i, 1000, x_test, y_test_one_hot.eval())
            a = accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})
            f_a += a

        print 'final accuracy : ', f_a/10

# train accuracy : above 85%
# test accuracy  : about 55%
#
# need to avoid overfitting
