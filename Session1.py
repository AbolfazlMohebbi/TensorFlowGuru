import tensorflow as tf
print("what is the TensorFlow version?   " + tf.__version__)

# Compute the mean of two numbers x and y using TensorFlow

# Defining the inputs using placeholders. These placeholders tell the program that we use these values later.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a graph: relationships
G_mean = tf.sqrt(x * y)
with tf.Session() as Sess:
    res = Sess.run(G_mean, feed_dict={x: 2.0, y: 8.0})
    print(res)


