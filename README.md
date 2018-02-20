Cifar10 classification using tensorflow
==========
0.0
----------
conv1 - max_pool1 - conv2 - max_pool2 -conv3 - conv4 - FC1 - FC2 - FC3

- set weight stddev: 0.01 <- why this work?
- too slow to train

0.1
----------
conv1 - max_pool1 - conv2 - max_pool2 -conv3 - conv4 - FC1 - FC2 - FC3

- add dropout(keep_prob: 0.5-0.7) FC1, FC2, FC3
- training accuracy : upto 85% with 20 epoch
- test accuracy : 53 - 55%

- low train/test accuracy
- overfitting (gap between training and test sets too large)
	
- try:
	+ data augmentation
	+ batch normalization
	+ add more dropouts

0.2
----------
conv1 - max_pool1 - conv2 - max_pool2 - conv3 - conv4 - conv5 - conv6 - max_pool3 - fc1 - fc2 - fc3

reference:
hunkim.github.io/ml
solarisailab.com
