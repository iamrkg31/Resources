"""Read/Write csv using pandas
Uses-
Nltk 3.2.4
Pandas 0.20.3
"""
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

# Import data
df1 = pd.read_csv("/home/rahul/sentiment_training_data.csv")

def tokenize_to_sentence(text, match1, match2):
    """Tokenizes to sentences"""
    sent_list = []
    for i in sent_tokenize(text):
        if re.search(match1.lower(), i.lower()) and re.search(re.sub("(\(|\))","",match2.lower()), i.lower()) :
            sent_list.append(i)
    return sent_list

# Add sentences to dataframe
df1["Sentences"] = df1.apply(lambda x:tokenize_to_sentence(x[0], x[2], x[3]), axis=1)
df2 = df1.apply(lambda x: pd.Series(x['Sentences']),axis=1).stack().reset_index(level=1, drop=True)
df2.name = "Sentence"
df1  = df1.drop(['Message', 'Sentences'], axis=1).join(df2)
df1 = df1[df1["Sentence"].notnull()]

# Write to file
df1.to_csv("out.csv", sep="\t")