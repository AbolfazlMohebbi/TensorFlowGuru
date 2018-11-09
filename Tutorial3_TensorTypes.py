import tensorflow as tf

# Create Constant Tensors
# tf.constant(value, dtype, name = "")
# - value: Value of n dimension to define the tensor. Optional
# - dtype: Define the type of data:
#     - tf.string: String variable
#     - tf.float32: Flot variable
#     - tf.int16: Integer variable
# - name: Name of the tensor. Optional.


matrix_2d = tf.constant([ [1, 2, 3],
                          [3, 4, 8] ], dtype = tf.int16, name = "my2Dmatrix")
print(matrix_2d)

matrix_3d = tf.constant([ [ [1, 1, 0, 3], [2.99, 8.9, 1.01, 9.99] ], [[5.2, 2.4, 2.1, 10], [3.22, 1.9, 8.32, 12.001] ] ], dtype = tf.float32, name = "my3Dmatrix")
print(matrix_3d)


# It is not possible to check the values of Tenors without running the graph.

#initialize the variable
init_op = tf.global_variables_initializer()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print(sess.run(matrix_2d))
    print(sess.run(matrix_2d[0,1]))  # row 0, col 1
    print(sess.run(matrix_2d[1,2]))  # row 1, col 2
    #print(sess.run(matrix_3d))
    print(sess.run(matrix_3d[1,0,1])) # mat 1, row 0, col 1
    print(sess.run(matrix_3d[0,1,:]))  # mat 0, row 1, col all
    print(sess.run(matrix_3d[1,:,:]))  # mat 1, row all, col all

# Following is a list of commonly used operations.
# tf.add(a, b)
# tf.substract(a, b)
# tf.multiply(a, b)
# tf.div(a, b)
# tf.pow(a, b)
# tf.exp(a)
# tf.sqrt(a)


# Create Variable Tensors
# tf.get_variable(name = "", shape, dtype, initializer)

MyInitializer = tf.constant ([ [4.2, 2.88] ],dtype = tf.float64, name = "myInitVar")

var1 = tf.get_variable("var1", [1, 2], dtype = tf.float64, initializer = tf.random_uniform_initializer)
var2 = tf.get_variable("var2", [1, 2], dtype = tf.float64, initializer = tf.zeros_initializer)
var3 = tf.get_variable("var3", dtype = tf.float64, initializer = MyInitializer)



# Create a placeholder
# A placeholder has the purpose of feeding the tensor.
# tf.placeholder(dtype,shape=None,name=None )

# refer to Tutorial1_Basics