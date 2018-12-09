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


# import the data
data = pd.read_csv("../data/iris/iris.csv")
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
# there are 4 features here, A's dimension is (4, 1)
A = tf.Variable(tf.random_normal(shape=[4, 1], dtype=tf.float64))
b = tf.Variable(tf.random_normal(shape=[1, 1], dtype=tf.float64))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# define placeholders
input = tf.placeholder(dtype=tf.float64, shape=[None, 4])
target = tf.placeholder(dtype=tf.float64, shape=[None, 1])

# declare the model you need to learn
hypothesis = tf.matmul(input, A) + b

# cost function
cost_fun = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=target))

# define the learning rate， batch_size etc.
learning_rate = 0.003
batch_size = 30
iter_num = 1500

# define the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)

# define goal
goal = opt.minimize(cost_fun)

# define the accuracy
# the default threshold is 0.5, rounded off directly
prediction = tf.round(tf.sigmoid(hypothesis))
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float64)
accuracy = tf.reduce_mean(correct)

# start training model
# define the variable that stores the result
loss_trace = []
train_acc = []
test_acc = []

for epoch in range(iter_num):
    # generate random batch index
    batch_index = np.random.choice(len(train_X), size=batch_size)
    batch_train_X = train_X[batch_index]
    batch_train_y = np.matrix(train_y[batch_index]).T
    sess.run(goal, feed_dict={input: batch_train_X, target: batch_train_y})
    temp_loss = sess.run(cost_fun, feed_dict={input: batch_train_X, target: batch_train_y})

    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={input: train_X, target: np.matrix(train_y).T})
    temp_test_acc = sess.run(accuracy, feed_dict={input: test_X, target: np.matrix(test_y).T})

    # recode the result
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

    # output
    if (epoch + 1) % 300 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,

                                                                          temp_train_acc, temp_test_acc))
# Visualization of the results
# loss function
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# accuracy
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()