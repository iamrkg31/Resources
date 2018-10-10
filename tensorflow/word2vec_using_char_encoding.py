"""Implementing an CNN in Tensorflow to predict spam/ham from texts using word2vec embedding
References-
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/#more-452
Uses -
python 3.6.3
tensorflow 1.10.0
numpy 1.14.5
pandas 0.20.3
matplotlib 2.1.0
gensim 3.3.0
"""
import re
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Start a graph
sess = tf.Session()

# Set RNN parameters
epochs = 100
batch_size = 250
max_word_length = 12
rnn_size = 15
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0001
filter_sizes = [3,4,5]
num_filters = 128
num_classes = 2

# Import data
data = pd.read_csv("data/spam.txt", delimiter="\t")

# Get all characters
def get_all_chars(texts):
    chars = set()
    for text in texts:
        chars = chars.union(set(list(text)))
    return sorted(list(chars))

# Get all words
def get_all_words(texts):
    words = set()
    for text in texts:
        text = re.sub("\W+", " ", text)
        words = words.union(set(text.split(" ")))
    return sorted(list(words))

# Get word embeddings
def get_word_embeddings(words, _max_word_length, chars_dict):
    for word in words:
        word_ids = np.zeros(_max_word_length, dtype=np.float32)
        for id, c in enumerate(list(word)):
            if id >= _max_word_length:
                break
            if c in chars_dict:
                word_ids[id] = chars_dict[c]
        yield word_ids

# Transform data
chars = get_all_chars(data["Mail"].tolist())
words = get_all_words(data["Mail"].tolist())

chars_dict = dict()
for id, c in enumerate(chars):
    chars_dict[c] = id

X = np.array(list(get_word_embeddings(words, max_word_length, chars_dict)))

input = tf.placeholder(tf.int32, [None, max_word_length])

# Create embedding
embedding_mat =  tf.get_variable(name="embedding_mat",dtype=tf.float32, shape=[len(chars), embedding_size])
embedding_output = tf.nn.embedding_lookup(embedding_mat, input)

# 3. bi lstm on chars
cell_fw = tf.contrib.rnn.LSTMCell(25, state_is_tuple=True)
cell_bw = tf.contrib.rnn.LSTMCell(25, state_is_tuple=True)

_, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
    cell_bw, embedding_output, dtype=tf.float32)

# shape = (batch x sentence, 2 x char_hidden_size)
output = tf.concat([output_fw, output_bw], axis=-1)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Start training
for epoch in range(epochs):
    num_batches = int(len(X) / batch_size) + 1

    for i in range(num_batches):
        # Select train data
        batch_index = np.random.choice(len(X), size=batch_size)
        batch_train_X = X[batch_index]

        train_dict = {input: batch_train_X}
        new_out = sess.run(output, feed_dict=train_dict)

        print(new_out.shape)
        print(new_out)