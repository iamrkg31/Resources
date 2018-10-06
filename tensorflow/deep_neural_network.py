"""Tensorflow implementation of logistic regression
Reference - https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow
Uses-
python 3.6.3
tensorflow 1.10.0
numpy 1.14.5
pandas 0.20.3
matplotlib 2.1.0
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Start a graph
sess = tf.Session()

# parameters
n_hidden_1 = 15            # 1st layer
n_hidden_2 = 15            # 2nd layer
n_hidden_3 = 15            # 3rd layer
n_hidden_4 = 15            # 4th layer
n_classes = 2               # output m classes
STDDEV = 0.1
epochs = 2000
batch_size = 250
learning_rate = 0.0001
num_classes = 2
dropout_keep_prob = tf.placeholder(tf.float32)

# import the data
data = pd.read_csv("data/iris.csv")
data = data[:100] # consider only 2 classes ie, upto 100 rows
data.Species = data.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])
X = data.drop(labels=["Id", "Species"], axis=1).values
y = data.Species.values

# set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# divide the inputs into train and test data
# 80% training data, 20% testing data
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]


def min_max_normalized(in_data):
    """Normalizes the input data"""
    col_max = np.max(in_data, axis=0)
    col_min = np.min(in_data, axis=0)
    return np.divide(in_data - col_min, col_max - col_min)

# normalize the training and testing data
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

# build the model framework
# declare the variables that need to be learned and initialization
# ie, weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([train_X.shape[1], n_hidden_1],stddev=STDDEV)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=STDDEV)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=STDDEV)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=STDDEV)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes],stddev=STDDEV)),
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def deep_neural_net(_X, _weights, _biases, _dropout_keep_prob):
    layer1 = tf.nn.dropout(tf.matmul(_X, _weights['h1']) + _biases['b1'], _dropout_keep_prob)
    layer2 = tf.nn.dropout(tf.matmul(layer1, _weights['h2']) + _biases['b2'], _dropout_keep_prob)
    layer3 = tf.nn.dropout(tf.matmul(layer2, _weights['h3']) + _biases['b3'], _dropout_keep_prob)
    layer4 = tf.nn.dropout(tf.matmul(layer3, _weights['h4']) + _biases['b4'], _dropout_keep_prob)
    out = tf.matmul(layer4, _weights['out']) + _biases['out']
    return out


# define placeholders
input = tf.placeholder(dtype=tf.float32, shape=[None, train_X.shape[1]])
target = tf.placeholder(dtype=tf.int32, shape=[None])

# declare the model you need to learn
logits_out = deep_neural_net(input, weights, biases, dropout_keep_prob)

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=target)
loss = tf.reduce_mean(losses)

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(target, tf.int64)), tf.float32))

# Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate)
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
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(train_X)))
    train_X = train_X[shuffled_ix]
    train_y = train_y[shuffled_ix]
    num_batches = int(len(train_X) / batch_size) + 1

    for i in range(num_batches):
        # Select train data
        batch_index = np.random.choice(len(train_X), size=batch_size)
        batch_train_X = train_X[batch_index]
        batch_train_y = train_y[batch_index]
        # Run train step
        train_dict = {input: batch_train_X, target: batch_train_y, dropout_keep_prob: 0.5}
        sess.run(goal, feed_dict=train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {input: test_X, target: test_y, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

print('\nOverall accuracy on test set (%): {}'.format(np.mean(temp_test_acc) * 100.0))

# Plot loss over time
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('RNN training/test loss')
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
