import tensorflow as tf

# operation output = xz + x^2 + 3z + 5

# using placeholders
x = tf.placeholder(tf.float32, name = "x")
z = tf.placeholder(tf.float32, name = "z")

output = tf.add(tf.multiply(x, z) + tf.pow(x, 2.0) + tf.multiply(3.0, z), 5.0)

with tf.Session() as sess1:
    result = sess1.run(output, feed_dict={x:[1, 0, 5], z:[2, 0, 6]})
    print(result)

# Using variables
xv = tf.get_variable("xv", dtype=tf.float32, initializer=tf.constant([1.0, 0.0, 5.0]))
zv = tf.get_variable("zv", dtype=tf.float32, initializer=tf.constant([2.0, 0.0, 6.0]))
pow = tf.constant([2.0])
constant = tf.constant([5.0])
coeff = tf.constant([3.0])

output2 = tf.add(tf.multiply(xv, zv) + tf.pow(xv, pow) + tf.multiply(coeff, zv), constant)

init = tf.global_variables_initializer()

with tf.Session() as sess2:
    init.run()
    result2 = output2.eval()
    print(result2)
