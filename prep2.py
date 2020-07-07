import os
import numpy as np
import matplotlib.pyplot as plt
import lda
import gensim
from gensim.corpora import Dictionary as dct
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import spacy
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
import os
from gensim.utils import simple_preprocess, lemmatize
import gensim
from nltk.corpus import stopwords
import re
import logging
from gensim import models
import BOW_creator
import nltk
#nltk.download()
from gensim.corpora import Dictionary as dct
import pyLDAvis.gensim
from evaluate import evaluate_graph
from gensim.models import CoherenceModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')
stop_words = stop_words + ['com', 'saw', 'could']





path = 'C:/Users/bramj/Desktop/Thesis/fairy-tales/tales/'

stemmer = SnowballStemmer("english")
# tokenize - break down each sentence into a list of words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def stopword_remover(text_raw):
    result = []
    for i in range(len(text_raw)):
        processed_docs = gensim.parsing.preprocessing.remove_stopwords(text_raw[i])
        result.append(processed_docs)
        print(processed_docs)
    return result

def shorting(text_raw):
    result = []
    for i in range(len(text_raw)):
        processed_sentence = gensim.parsing.preprocessing.strip_short(text_raw[i], minsize=3)
        result.append(processed_sentence)
        print(processed_sentence)
    return result

def stemming(text_raw):
    result = []
    for i in range(len(text_raw)):
        processed_docs = gensim.parsing.preprocessing.stem_text(text_raw[i])
        result.append(processed_docs)
        print(processed_docs)
    return result




data = []
counter = 0
dictionary = dct(data)
bow_corpus = []
bow_dict = dct()
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        counter += 1
        op = open(path + name, 'r')
        text_raw = op.readlines()
        dat = gensim.parsing.preprocessing.preprocess_documents(text_raw)
        print(dat)
        data.append(dat)
        dictionary.add_documents(dat)
        bow_corpi = [dictionary.doc2bow(doc) for doc in dat]
        bow_corpus += bow_corpi

iterator =0
for list in data:
    for doc in list:
        print(iterator)
        iterator +=1
        print(doc)


print('Number of documents is: ', counter)
print(" \n num_docs: (Number of documents processed.)", dictionary.num_docs)
print('Dictionary length is:', len(dictionary))
print("\n Processed words:" , dictionary.num_pos)

print("Data is: ", data)

print("Non zeros in dict: ",dictionary.num_nnz)
bow_doc_10 = bow_corpus[10]
bow_doc_20 = bow_corpus[20]
print(bow_corpus[10])
print(bow_corpus[20])

for i in range(len(bow_doc_10)):
  print(f'Word {bow_doc_10[i][0]} (\"{dictionary[bow_doc_10[i][0]]}\") appears {bow_doc_10[i][1]} time')

for i in range(len(bow_doc_20)):
   print(f'Word {bow_doc_20[i][0]} (\"{dictionary[bow_doc_20[i][0]]}\") appears {bow_doc_20[i][1]} time')


#model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=10)
print("REACHED: MADE MODEL")

#for idx, topic in model.print_topics(-1):
    #print("Topic: {} \nWords: {}".format(idx, topic))
    #print("\n")

#lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=bow_corpus, texts=data, limit=10)


#cm = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence='c_v')

c_v = []
lm_list = []
for num_topics in range(1, 3):
    lm = LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary)
    lm_list.append(lm)
    cm = CoherenceModel(model=lm, texts=data, dictionary=dictionary, coherence='c_v')
    c_v.append(cm.get_coherence())

#print(dictionary.token2id)

x = range(1, 3)
plt.plot(x, lm_list)
plt.xlabel("num_topics")
plt.ylabel("Coherence score")
plt.legend(("c_v"), loc='best')
plt.show()