import math as m
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import sklearn.metrics as skm

## Database Collection
docs = {
    "d1" : "scary green crocodile",
    "d2" : "scary green big",
    "d3" : "small crocodile",
}
#### Data Preprocessing
def tokenizer(text):
    return text.split()
tokenized_docs = {}
for d, doc in docs.items():
    tokenized_docs[d] = tokenizer(doc)
#print(tokenized_docs)

all_words = []
for doc in tokenized_docs.values():
    for word in doc:
        all_words.append(word)              ## Collect All Words
print("All words: ",all_words)
unique_words = set(all_words)               ## Remover duplicate Words
print("Unique words: ",unique_words)
vocab = sorted(list(unique_words))          ## Sort alphabetically
print("Sorted Unique words: ",vocab)

#### Calculate Term Frequency TF Matrix
tf= {}
for d, terms in tokenized_docs.items():
    counter = Counter(terms)
    #print(counter)
    tf_row = []
    for word in vocab:
        if word in counter:
            tf_row.append(counter[word])
        else:
            tf_row.append(0)
    tf[d] = tf_row
print("\nTF Matrix (rows=docs, columns=terms):")
for d in docs:
    print(d, tf[d])

#### Calculate DF(Document Frequency) and IDF (Inverse Document Frequency)
df = {}
for terms in vocab:
    counter = 0;
    for d in docs:
        words_in_doc = tokenized_docs[d]
        if terms in words_in_doc:
            counter += 1
        df[terms] = counter

N= len(docs)
idf = {}
for terms in vocab:
    idf[terms] = m.log10(N/df[terms])

print("\nIDF Matrix (rows= Terms, columns= IDF):")
for terms in idf:
    print(terms, round(idf[terms],3))

#### Calculate TF-IDF (Term-concept association) Matrix
tfidf = {}
for d in docs:
    tfidf_row = []
    for i in range(len(vocab)):
        term = vocab[i]
        tfidf_weight = tf[d][i] * idf[term]
        tfidf_row.append(round(tfidf_weight,3))
    tfidf[d] = tfidf_row

print("\nTF-IDF Matrix (rows=docs, columns=TFIDF):")
for terms in tfidf:
    print(terms, tfidf[terms])

#### Calculate ESA Vector
def esa_vec(text):
    words = tokenizer(text)
    t_tf = Counter(words)
    t_vec = []
    for term in vocab:
        if term in t_tf:
            t_vec.append(t_tf[term] * idf[term])
        else:
            t_vec.append(0)
    ##print(t_vec)
    ## Now Calculate concept score, one score for per document
    concept_score = []
    for d in docs:
        score = sum(t_vec[i] * tfidf[d][i] for i in range(len(vocab)))
        concept_score.append(score)

    return concept_score


#### Cosine Similarity Calculation
def cosine_sim(v1, v2):
    return  np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#### Question 1
T = "green crocodile"
V = esa_vec(T)
print("\n ESA Vector for 'green crocodile' :",np.round(V,3))

#### Question 2
v1 = "big crocodile"
v2 = "scary crocodile"
v1_esa_vec = esa_vec(v1)
v2_esa_vec = esa_vec(v2)

similarity = cosine_sim(v1_esa_vec, v2_esa_vec)

print("\n ESA Vector for 'big crocodile' :",np.round(v1_esa_vec,3))
print("\n ESA Vector for 'scary crocodile' :",np.round(v2_esa_vec,3))
print("\n Cosine Similarity: ",np.round(similarity,3))