import tensorflow as tf

# Step 1: Define the variables
X_1 = tf.placeholder(tf.float32, name = "X_1")
X_2 = tf.placeholder(tf.float32, name = "X_2")

# Define the operation
multiply = tf.multiply(X_1, X_2, name = "multiply")

# Create a Chunk (Session) to run the operation
with tf.Session() as MultiplySession:
    result = MultiplySession.run(multiply, feed_dict={X_1:[1, 9, 8], X_2:[3, 2, 4]})
print(result)

# Notes:
# 1 - the "feed_dict" word is mandatory
# 2 - the inputs should be separated with "," not spaces. this is not Matlab


# Following is a list of commonly used operations.
# tf.add(a, b)
# tf.substract(a, b)
# tf.multiply(a, b)
# tf.div(a, b)
# tf.pow(a, b)
# tf.exp(a)
# tf.sqrt(a)


