"""Python code to perform text preprocessing"""
import re
import html
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# List of contractions
contraction_list = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}


def filter_stopwords(doc):
    """Removes stpwords"""
    doc = ' '.join([word for word in doc.split() if word not in stopwords.words("english")])
    #print (doc.split())
    return doc


def lemma(doc):
    """Performs lemmatization"""
    wordnet_lemmatizer = WordNetLemmatizer()
    doc = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in doc.split()])
    doc = ' '.join([wordnet_lemmatizer.lemmatize(word,'v') for word in doc.split()])
    return doc


def escape_html_chars(text):
    """Converts html special codes"""
    return html.unescape(text)


def expand_contractions(text):
    """Expands contractions"""
    words = text.split()
    modified_text = [contraction_list[word] if word in contraction_list else word for word in words]
    modified_text = " ".join(modified_text)

    return modified_text


def replace_slangs(text,slang_dict):
    """Replaces slangs to standard words"""
    words = text.split()
    modified_text = [slang_dict[word] if word in slang_dict else word for word in words]
    modified_text = " ".join(modified_text)

    return modified_text


def filter_links(text):
    """Removes links from the data"""
    return re.sub(r'(http(s)?:\/\/\S*?( |$)|www\.\S*?( |$))', "", text)


def create_slang_dict():
    """Creates slang dictionary from a text file"""
    slang_dict = {}
    slang_dict_path = "slangs.txt"

    with open(slang_dict_path) as file:
        for line in file:
            if not line:
                continue
            line = line.lower()
            row = line.split(":")
            key = row[0].strip()
            value = row[1].strip()
            slang_dict[key]=value

    return slang_dict


def main():
    file_path_to_read = "input.txt"
    file_path_to_write = "out.txt"
    slang_dict = create_slang_dict()
    f = open(file_path_to_write, 'w')
    count = 0
    with open(file_path_to_read) as file:
        for line in file:
            if count == 0:
                count = count + 1
                continue
            count = count + 1
            print(count)
            # ignore messages with less than or equal to 5 words
            if not line or len(line.split()) <= 5:
                continue
            line = escape_html_chars(line)
            line = filter_links(line)
            line = line.lower()
            line = replace_slangs(line,slang_dict)
            line = expand_contractions(line)
            line = re.sub('\W+', ' ', line)
            line = filter_stopwords(line)
            f.write(line + '\n')
    f.close()


if __name__ == "__main__":
    main()