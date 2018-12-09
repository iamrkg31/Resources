"""Implementing an RNN in Tensorflow to predict spam/ham from texts using vocab lookup
References-
Predictive Analytics with TensorFlow - chapter 9
Uses -
python 3.6.3
tensorflow 1.10.0
numpy 1.14.5
pandas 0.20.3
matplotlib 2.1.0
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 250
batch_size = 250
max_sequence_length = 25
rnn_size = 15
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0001
dropout_keep_prob = tf.placeholder(tf.float32)

# Import data
data = pd.read_csv("../data/spam/spam.txt", delimiter="\t")

# Clean text
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return (text_string)

# Transform data
data["Mail"] = data.apply(lambda x:clean_text(x[1]), axis=1)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
X = np.array(list(vocab_processor.fit_transform(data["Mail"].tolist())))
data.Class = data.Class.replace(to_replace=['ham', 'spam'], value=[0, 1])
y = data.Class.values

# Set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# Divide the inputs into train and test data
# 80% training data, 20% testing data
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]

# Create placeholders
input = tf.placeholder(tf.int32, [None, max_sequence_length])
target = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, input)

# Define the RNN cell
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
output, state = tf.nn.dynamic_rnn(basic_cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)
last = output[:,-1,:]

# Create weight and bias variable
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))

# Create logits
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

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
    num_batches = int(len(train_X)/batch_size) + 1

    for i in range(num_batches):
        # Select train data
        batch_index = np.random.choice(len(train_X), size=batch_size)
        batch_train_X = train_X[batch_index]
        batch_train_y = train_y[batch_index]
        # Run train step
        train_dict = {input: batch_train_X, target: batch_train_y, dropout_keep_prob:0.5}
        sess.run(goal, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {input: test_X, target: test_y, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))

print('\nOverall accuracy on test set (%): {}'.format(np.mean(temp_test_acc)*100.0))  
 
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
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
