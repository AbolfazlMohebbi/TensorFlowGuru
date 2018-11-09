import numpy as np
# Python List
myPythonList = [1,9,8,3]

#NumPy Array
npVector = np.array([1,9,8,3])

npMatrix2 = np.array([(1,2,3,4),
                      (5,6,7,8)])

npMatrix2_2 = np.array([[1,2,3,4],
                        [5,6,7,8]])

npMatrix3 = np.array([ [[1, 2, 3, 4], [5, 6, 7, 8]], [[0, 1, 0, 19], [12, 19, 10, 18]]])

print(npVector); print(npVector.shape)
print(npMatrix2); print(npMatrix2.shape)
print(npMatrix2_2); print(npMatrix2_2.shape)
print(npMatrix3); print(npMatrix3.shape)

# Zeros and Ones
npZerosExample = np.zeros((5,5), dtype=np.float64)
npOnesExample  = np.ones((3,2), dtype=np.int64)
print(npZerosExample); print(npZerosExample.shape)
print(npOnesExample); print(npOnesExample.shape)


# Reshape
npMatrix2_rs = npMatrix2.reshape(4,2)
print(npMatrix2_rs)
print(npMatrix2_rs.shape)

npMatrix2_flat = npMatrix2.flatten()
print(npMatrix2_flat)
print(npMatrix2_flat.shape)

# Stacking matrices
stMatrix1 = np.array([(0,-2,-3),
                      (15,43,17)])
stMatrix2 = np.array([(1,2,3),
                      (5,6,7)])

stackV = np.vstack((stMatrix1, stMatrix2))
stackH = np.hstack((stMatrix1, stMatrix2))
print(stackV)
print(stackH)

# Generate Random Numbers

# numpy .random.normal(loc, scale, size)
# Loc: the mean. The center of distribution
# scale: standard deviation.
# Size: number and size of returns

normal_array = np.random.normal(5, 0.5, (2,5))
print(normal_array)

# Modifying Arrays
Matrix1 = np.matrix([(0,-2,-3),
                      (15,43,19)])
print(Matrix1)
print(Matrix1[1,2])
# Next 2 lines are the same
print(Matrix1[1,:])
print(Matrix1[1])
Matrix1[1,2] = 17
print(Matrix1[1,2])


# Matrix Operations
sqMatrix1 = np.array([(1,2,3),
                      (5,6,7),
                      (8,9,10)])

sqMatrix2 = np.array([(0,2,-3),
                      (5,12,1),
                      (-1,0,10)])


print('Matrix Multiply:', np.matmul(sqMatrix1, sqMatrix2))

print('Element Wise Product', sqMatrix1*sqMatrix2)
print('Element Wise Product2', np.multiply(sqMatrix1, sqMatrix2))







