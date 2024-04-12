import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokenize(setentance):
    return nltk.word_tokenize(setentance)
def stem(word):
    return stemmer.stem(word,to_lowercase=True)
def back_of_word(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1

    return bag


# if __name__=="__main__":
#     pass