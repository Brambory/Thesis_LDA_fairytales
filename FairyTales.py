import dictionary as dictionary
import gensim
import pprint
import math
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from gensim import models
from gensim import similarities
from collections import defaultdict
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize

data = open('C:/Users/bramj/Desktop/Thesis/fairy-tales/merged_clean.txt', 'r')

text_raw = data.readlines()

print("Current length raw text pre- stoplist filter" ,len(text_raw))

# TD: Currently stoplist filtering doesn't do shit



stoplist = set('for a of the and to in the they he she his that this is have had'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_raw]

print("Current length raw text after- stoplist filter",len(texts))

stemmer = SnowballStemmer("english")
# Build the bigram and trigram models
bigram = Phrases(texts, min_count=3, threshold=80) # higher threshold fewer phrases.
trigram = Phrases(bigram[texts], threshold=80)# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)# See trigram example
print(trigram_mod[bigram_mod[texts[100]]])



#def remove_stopwords(texts):
  # return [[word for word in gensim.utils.simple_preprocess()simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# perform lemmatiziatian (words changed to first person and present tense) and
# stemming (words to root form)
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# perform preprocessing steps on dataset
# e.g. remove word if in stopwords or length <=3
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # print(f'token is: {token}')
            result.append(lemmatize_stemming(token))
            # print(f'result is: {result}')
    return result


#preprocess(texts)
"""""""""""
Preprocess doesnt currently work on my text sets
"""""""""""

dictionary = gensim.corpora.Dictionary(texts)
data.close()
print("Current length raw text after- lemmatization/ preprocess filter",len(texts))
print("Length Dict Pre-preprocess:",len(dictionary))
print("Reached End Preprocessing")

counter = 0
for key, value in dictionary.iteritems():
    print(key, value)
    counter += 1
    if counter > 10:
        break

#dictionary.filter_n_most_frequent(2000)
""""""""""
Using only this filter makes this better, but still shit on it's own.

"""""""""""
#dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=100000)
#print('he' in dictionary.token2id)
#dictionary.filter_tokens(bad_ids=[dictionary.token2id['he']])
#print('he' in dictionary.token2id)
"""""""""""
TD: IT APPEARS THAT THE DICTIONARY.FILTER_TOKENS DOESNT DO ANYTHING RIGHT NOW
FURTHERMORE: RECHECK WHICH TOKENS NEED TO BE FILTERED
"""""""""""



print("Current Length Dict:", len(dictionary))
print("Reached End Filtering")

#rint(dictionary.dfs)
bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
print(bow_corpus[1000])

bow_doc_1000 = bow_corpus[1000]

for i in range(len(bow_doc_1000)):
    print(f'Word {bow_doc_1000[i][0]} (\"{dictionary[bow_doc_1000[i][0]]}\") appears {bow_doc_1000[i][1]} time')

print(bow_corpus[2000])
bow_doc_2000 = bow_corpus[2000]

for i in range(len(bow_doc_2000)):
    print(f'Word {bow_doc_2000[i][0]} (\"{dictionary[bow_doc_2000[i][0]]}\") appears {bow_doc_2000[i][1]} time')


tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=50)
print("REACHED: MADE MODEL")

for idx, topic in model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

print(dictionary)
unseen_document = "There was a little Frog Whose home was in a bog, And he worried 'cause he wasn't big enough. He sees an ox and cries: That's just about my size, If I stretch myself--Say Sister, see me puff! So he blew, blew, blew, Saying: Sister, will that do? But she shook her head. And then he lost his wits. For he stretched and puffed again Till he cracked beneath the strain, And burst, and flew about in little bits."

bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(model[bow_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, model.print_topic(index, 5)))

"""""""""
#index = similarities.SparseMatrixSimilarity(model[bow_corpus], num_features=12)

index = similarities.SparseMatrixSimilarity(model[corpus_tfidf], num_features=12)

query = "Birch Fox Kangaroo".split()
query_bow = dictionary.doc2bow(query)
sims = index[model[query_bow]]

print("Similarity query to corpus:")
#print(list(enumerate(sims)))

#print("Sorted enum:")
#for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
#    print(document_number, score)
"""""""""
print("reached end")

# Visualize the topics
#pyLDAvis.enable_notebook(sort=True)
#vis = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary)
#pyLDAvis.display(vis)

print("reached vis")