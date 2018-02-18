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
W1 = tf.Variable(tf.random_normal([11, 11, 3, 96]), tf.float32)
L1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#L1.shape=[None, 16, 16, 96] 

#L2
W2 = tf.Variable(tf.random_normal([5, 5, 96, 256]), tf.float32)
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#L2.shape=[None, 8, 8, 256]

#L3
W3 = tf.Variable(tf.random_normal([3, 3, 256, 384]), tf.float32)
L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.relu(L3)
#L3.shape=[None, 8, 8, 384] 

#L4
W4 = tf.Variable(tf.random_normal([3, 3, 384, 384]), tf.float32)
L4 = tf.nn.conv2d(L3, W4, strides = [1, 1, 1, 1], padding = 'SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
#L4.shape=[None, 4, 4, 384]

#FC1
L4 = tf.reshape(L4, [-1, 4*4*384])
W5 = tf.Variable(tf.random_normal([4*4*384, 384]))
b5 = tf.Variable(tf.random_normal([384]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

#FC2
W6 = tf.Variable(tf.random_normal([384, 384]))
b6 = tf.Variable(tf.random_normal([384]))
L6 = tf.nn.relu(tf.matmul(L5,W6) + b6)

#FC3
W7 = tf.Variable(tf.random_normal([384, 10]))
b7 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L6,W7) + b7
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

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
            h, l, _ = sess.run([accuracy, loss, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            if step%10 == 0:
                print 'step : ', step, 'accuracy : ', h

        print 'epoch : %04d' %(epoch+1), 'cost : ', l, 'accuracy : ', accuracy.eval(feed_dict={X:batch_xs, Y:batch_ys})

    #sample
    rand = random.randrange(0, 10000)
    
    test_xs = x_test[rand, :]
    test_ys = int(y_test[rand, :])
        
    h = tf.squeeze(sess.run(hypothesis, feed_dict={X:[test_xs]}))
    p = tf.argmax(h, axis = 0)
    p = tf.cast(p, tf.int32)
    print 'predicted : ', labels[p.eval()] 
    print 'answer : ', labels[test_ys]

    plt.imshow(test_xs)
    plt.show()
