"""Tensorflow implementation of deep neural network
Reference -
Deep Learning with TensorFlow: Applications of Deep Neural Networks to Machine Learning Tasks - chapter 4
Uses-
python 3.6.3
tensorflow 1.10.0
numpy 1.14.5
matplotlib 2.1.0
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Start a graph
sess = tf.Session()

# parameters
n_input = 784
n_hidden_1 = 64            # 1st layer
n_hidden_2 = 64            # 2nd layer
n_classes = 10               # output m classes
STDDEV = 0.1
epochs = 50
batch_size = 128
learning_rate = 0.1
dropout_keep_prob = tf.placeholder(tf.float32)

# import the data
mnist = input_data.read_data_sets("../data/MNIST", one_hot=True)

# Hidden layer with ReLU activation
def layer(x, W, b):
    z = tf.add(tf.matmul(x, W), b)
    a = tf.nn.relu(z)
    return a

# Neural net
def ann(x, _weights, _biases, _dropout_keep_prob):
    layer1 = layer(x, weights['W1'], biases['b1'])
    layer2 = layer(layer1, weights['W2'], biases['b2'])
    out = tf.add(tf.matmul(layer2, weights['W_out']), biases['b_out'])
    return out

# weights and biases
weights = {
    'W1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=STDDEV)),
    'W2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=STDDEV)),
    'W_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],stddev=STDDEV))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_out': tf.Variable(tf.random_normal([n_classes]))
}

# define placeholders
input = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
target = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

# declare the model you need to learn
logits_out = ann(input, weights, biases, dropout_keep_prob)

# Loss function
losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits_out, labels=target)
loss = tf.reduce_mean(losses)

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.argmax(target, 1)), tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
goal = optimizer.minimize(loss)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Define the variable that stores the result
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Start training
for epoch in range(epochs):
    num_batches = int(mnist.train.num_examples / batch_size) + 1
    for i in range(num_batches):
        batch_train_X, batch_train_y = mnist.train.next_batch(batch_size)
        # Run train step
        train_dict = {input: batch_train_X, target: batch_train_y, dropout_keep_prob: 0.5}
        sess.run(goal, feed_dict=train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {input: mnist.test.images, target: mnist.test.labels, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

print('\nOverall accuracy on test set (%): {}'.format(np.mean(temp_test_acc) * 100.0))

# Plot loss over time
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('training/test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
