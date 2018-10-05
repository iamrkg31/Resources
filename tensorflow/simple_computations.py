"""Simple computations in tensorflow
References -
Learning Path: TensorFlow: The Road to TensorFlow Second Edition - chapter 3
Uses-
python 3.6.3
tensorflow 1.10.0
"""

import tensorflow as tf

# Create constants to hold specific values
a = tf.constant(1)
b = tf.constant(2)

# Conputations on TF scalars
c = a + b
d = a * b

# TF numbers are stored in tensors
# A fancy term for multidimentional arrays
v1 = tf.constant([1., 2.]) # 1d vector
v2 = tf.constant([3., 4.]) # 2d vector
m = tf.constant([[1., 2.]]) # 2d matrix
n = tf.constant([[1., 2.], [3., 4.]]) # 2d matrix
k = tf.constant([[[1., 2.],[3., 4.]]]) # 3d+ matrix

# Computations on TF tensors
# Be careful of shapes
v3 = v1 + v2

# Operations are element-wise by default
m2 = m * m

# True matrix multiplication requires a special call
nn = tf.matmul(n,n)

# The above code only defines TF graph
# Nothing has been computed yet
# For that, we first need to create a TF session
sess = tf.Session()

# Now we can run the specific nodes of the graph
# ie, the variables, we have named
output = sess.run(nn)
print("nn is :")
print(output)

# Remember to close the session
# When done using it
sess.close()

# Often , we work interactively
# it's convenient to use a simplified session
sess = tf.InteractiveSession()

# Now we can compute any node
print("m2 is :")
print(m2.eval())

# TF variables can change value
# useful for updating model weights
w = tf.Variable(0, name="weight")

# But variables must be initialized by TF before use
init_op = tf.global_variables_initializer()
sess.run(init_op)

print("w is:")
print(w.eval())

w += a
print("w after adding a:")
print(w.eval())

w += a
print("w after adding a again:")
print(w.eval())

# We can return or supply arbitrary nodes
# ie, check an intermmediate value or
# sub any value in th middle of a computation
e = d + b
print("e is :")
print(e.eval())

# Lets see what d was at the same time
print("e and d :")
print(sess.run([e, d]))

# Use a custom d by specifying a dictionary
print("e with custom d=4 :")
print(sess.run(e, feed_dict = {d:4.}))