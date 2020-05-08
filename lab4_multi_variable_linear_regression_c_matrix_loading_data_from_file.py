import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]  # from the fist one, exclude the last one
y_data = xy[:, [-1]]  # only the last one

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

print("\n\n")

# Python List slicing
nums = list(range(5))
print (nums)
print (nums[2:4])
print (nums[2:])
print (nums[:2])
print (nums[:])
print (nums[:-1])
nums[2:4] = [8, 9]
print (nums)

print("\n\n")

# Numpy Indexing, Slicing, Iterating
a = np.array([1, 2, 3, 4, 5])  # array([1, 2, 3, 4, 5])
print (a[1:3])  # array([2, 3])
print (a[-1])   # 5
a[0:2] = 9
print (a)       # array([9, 9, 3, 4, 5])

print("\n\n")

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# array([[1, 2, 3, 4]
#        [5, 6, 7, 8]
#        [9, 10, 11, 12]])
print (b[:, 1])      # array([ 2, 6, 10])
print (b[-1])        # array([ 9, 10, 11, 12]) 
print (b[-1, :])     # array([ 9, 10, 11, 12])
print (b[-1, ...])   # array([ 9, 10, 11, 12])
print (b[0:2, :])    # array([1, 2, 3, 4], [5,6,7,8])
print (b[0:2])       # array([1, 2, 3, 4], [5,6,7,8]) 

print("Done")
