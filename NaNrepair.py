from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.phrases import Phrases, Phraser
import gensim
import os
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import strip_short
from gensim.corpora import Dictionary as dct


path = 'C:/Users/bramj/Desktop/Thesis/fairy-tales/tales/'

op = open(path + 'aunt.txt', 'r')
text_raw = op.readlines()


lemmatizer =WordNetLemmatizer()
print(lemmatizer.lemmatize("beetles"))
stemmer = SnowballStemmer("english")



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

def prep(text_raw):
    result = []
    for i in range(len(text_raw)):
        proc = gensim.parsing.preprocessing.preprocess_documents(text_raw[i])
        print(proc)
        result.append(proc)
    return result


data5 = gensim.parsing.preprocessing.preprocess_documents(text_raw)
print("REGULAR PREP: " , data5)
dictionary = dct(data5)

#dictionary.add_documents(dat)

bow_corpus = [dictionary.doc2bow(doc) for doc in data5]

print("Bow_corpus: ", bow_corpus)



path = 'C:/Users/bramj/Desktop/Thesis/fairy-tales/tales/'

op = open(path + 'blue beard.txt', 'r')
text_raw = op.readlines()

data6 = gensim.parsing.preprocessing.preprocess_documents(text_raw)

#bow = [dictionary.doc2bow(doc) for doc in data6]

bow = [dictionary.doc2bow(doc, allow_update=True) for doc in data6]
print("Bow6: ", bow)



bow_doc_10 = bow[10]

for i in range(len(bow_doc_10)):
    print(f'Word {bow_doc_10[i][0]} (\"{dictionary[bow_doc_10[i][0]]}\") appears {bow_doc_10[i][1]} time')
#bow_corpus.append(bow)

#print(bow_corpus)

print("DICT TEST1: ", dictionary)
gensim.corpora.dictionary.Dictionary.compactify(dictionary)

print("DICT TEST2: ", dictionary)

sentence = "walk walking walker walked to walk have been walking"
print(stemming(sentence))

print(shorting(sentence))

print(stopword_remover(sentence))




data4 = shorting(text_raw)
print("DOES SHORTENING WORK? FIND OUT NOW: ", data4 )




data1 = stopword_remover(text_raw)

print("THIS IS STOPWORD REMOVAL:" , data1)

data2 = stemming(data1)

#print("THIS IS STEMMING: ", data2)

print("THIS IS STOPWORD + STEMMING:" , data2)


data3 = shorting(data2)

print("THIS IS SPARTA!! NO EKSUALLY SHORTING:" , data3)

print("LENGTH RAW: ", len(text_raw), '\n' "LENGTH END: ", len(data3))





