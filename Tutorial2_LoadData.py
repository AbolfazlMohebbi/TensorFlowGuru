# There is two commons way to load data:
# 1. Load data into memory (Pandas library): Small Data i.e. 1-5 GB or less
# 2. Tensorflow data pipeline: Large data 10GB and up

import numpy as np
import tensorflow as tf

x_input = np.random.sample((4,3))  # 4 batches of data with 3 values
print(x_input)
x = tf.placeholder(tf.float32, shape=[4,3], name="inputX") # create a placeholder for 2 batches of data with 3 values

# define the Dataset where we can populate the value of the placeholder x
dataSet = tf.data.Dataset.from_tensor_slices(x)

# Initialize a pipeline where the data will flow in sequences
iterator = dataSet.make_initializable_iterator()
get_next = iterator.get_next() # a method to get the next batch of data

with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iterator.initializer, feed_dict={ x: x_input })
    print(sess.run(get_next)) # Print batch 1
    print(sess.run(get_next)) # Print batch 2
    print(sess.run(get_next)) # Print batch 3
    print(sess.run(get_next)) # Print batch 4

