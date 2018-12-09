"""Word2vec using gensim
Uses -
gensim 3.3.0
"""
import gensim

# Load word2vec/glove vector from text file
# file format -
# 2 50
# word1 -0.38497   0.80092   0.064106 ................  -0.28355
# word2 -0.33722   0.53741  -1.0616   ................ -0.081403
model_path = "../models/glove.6B.50d.txt"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=False)

print(model["hello"])