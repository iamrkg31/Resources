"""Implementing word2vec embedding using character embedding
References-
https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
Uses -
python 3.6.3
tensorflow 1.10.0
numpy 1.14.5
pandas 0.20.3
"""
import re
import numpy as np
import pandas as pd
import tensorflow as tf

# Start a graph
sess = tf.Session()

# Set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# Parameters
max_word_length = 12
embedding_size = 50
cell_size = 50 # dimentionality of hidden state/output
batch_size = 150
n_epochs = 100

# Import data
data = pd.read_csv("../data/spam/spam.txt", delimiter="\t")

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

# Get all words and chars
chars = get_all_chars(data["Mail"].tolist())
words = get_all_words(data["Mail"].tolist())

chars_dict = dict()
for id, c in enumerate(chars):
    chars_dict[c] = id

# Input
X = np.array(list(get_word_embeddings(words, max_word_length, chars_dict)))
input = tf.placeholder(tf.int32, [None, max_word_length])

# Create embedding
embedding_mat =  tf.get_variable(name="embedding_mat",dtype=tf.float32, shape=[len(chars), embedding_size])
embedding_output = tf.nn.embedding_lookup(embedding_mat, input) # shape = (batch, max_word_length, embedding_size)

# Bi lstm
cell_fw = tf.contrib.rnn.LSTMCell(cell_size, state_is_tuple=True)
cell_bw = tf.contrib.rnn.LSTMCell(cell_size, state_is_tuple=True)
_, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                      embedding_output,
                                                                      dtype=tf.float32)

# shape = (batch, 2 x cell_size)
output = tf.concat([output_fw, output_bw], axis=-1)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Get word embedding
word_embedding = np.zeros(shape=(len(words),2*cell_size))
for epoch in range(n_epochs):
    num_batches = int(len(X) / batch_size) + 1
    print("itr: ",epoch)
    for i in range(num_batches):
        batch_index = np.random.choice(len(X), size=batch_size)
        batch_train_X = X[batch_index]
        train_dict = {input: batch_train_X}
        word_embedding[batch_index] = sess.run(output, feed_dict=train_dict)

# Write word embedding to file
final_embeddings = pd.DataFrame(word_embedding,dtype=str)
final_embeddings.insert(loc=0, column="words", value=words)
final_embeddings.to_csv("../data/out.csv", sep=" ", index=False, header=False)