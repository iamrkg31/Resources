"""Read/Write csv using pandas"""
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

# Import data
df1 = pd.read_csv("/home/rahul/train.csv")
checks = pd.read_csv("/home/rahul/check_list.csv")["checks"].tolist()
checks_regex = "("+'|'.join(checks)+")"

def tokenize_to_sentence(text):
    """Tokenizes to sentences"""
    sent_list = []
    for i in sent_tokenize(text):
        if re.search(checks_regex.lower(),re.sub("\W+"," ",i.lower())):
            sent_list.append(i)
    return sent_list

# Add sentences to dataframe
df1["Sentences"] = df1.apply(lambda x:tokenize_to_sentence(x[2]), axis=1)
df2 = df1.apply(lambda x: pd.Series(x['Sentences']),axis=1).stack().reset_index(level=1, drop=True)
df2.name = "Sentence"
df1  = df1.drop(['Text', 'Sentences'], axis=1).join(df2)

# Write to file
df1.to_csv("out.csv", sep="\t")