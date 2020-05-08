#!/usr/bin/env python

import tensorflow as tf
tf.__version__

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, Tensorflow!")

# seart a TF session
sess = tf.Session()

# run the op and get result
print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)
print("node3: ", node3)

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + profices a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))

# X and Y data
# H(x)=Wx + b
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Our hypothesis XW+b
hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize Cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

print("Done")

# X and Y data
# H(x)=Wx + b
#x_train = [1,2,3]
#y_train = [1,2,3]

# Now we can use X and Y in place of x_data and y_data
# placeholders for a tensor that will be always fed using feed_dict

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Our hypothesis XW+b
hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize Cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ =     sess.run([cost, W, b, train], feed_dict={X: [1,2,3], Y:[1,2,3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print("Done")
