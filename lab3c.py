import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weight

W = tf.Variable(-5.0)

# Linear model
hypothesis = X * W

# Manual gradient using differential
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Get gradients
gvs = optimizer.compute_gradients(cost)

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(20):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

print("Done")
