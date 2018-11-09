import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

#This gives info about the Tensor node result not the actual results
print(result)

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(result)
  print(output)