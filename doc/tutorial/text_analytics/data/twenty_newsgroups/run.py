#!/ilab/users/sfj19/anaconda3/bin/python
""" Script to test my shit """

# imports
#import math
import random
from collections import defaultdict
from numpy import dot, array
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# constants
TWENTY_TRAIN = fetch_20newsgroups()
COUNT_VECT = CountVectorizer()
TFIDF_TRANSFORMER = TfidfTransformer()
N_NEWSGROUPS = len(TWENTY_TRAIN.target_names)
ONE_OUT_OF = 10000


# partition newsgroups
DOCUMENT_PARTITION = [[] for x in range(N_NEWSGROUPS)]
for i in range(len(TWENTY_TRAIN.target)):
    t = TWENTY_TRAIN.target[i]
    DOCUMENT_PARTITION[t].append(i)


def cos_sim(one, one_id, two, two_id, norms):
    """ Calculates cosime similarity between two vectors of same length """
    return dot(one, two)/(norms[one_id]*norms[two_id])


def get_row_norms(matrix):
    """ Calculates magnitudes for each row in a matrix. """
    norms = []
    for row_num in range(matrix.shape[0]):
        row = matrix.getrow(row_num).toarray()[0]
        norms.append(norm(row))
    return norms


# set up matrices, norms, and final scores for each data representation method
print('Building bag of words...')
WORDS = set()
DELIMITERS = '/\\\'\"!@#$%^&*()-+{}[]:;.,~`\t\n<>?=_'

ROW = []
COL = []
DATA = []
LOOKUP = defaultdict(int)
POSITIONS = defaultdict(int)

# obtain set of unique words
for datum in TWENTY_TRAIN.data:
    text = ' '.join(datum.split('\n')).split(' ')
    realtext = [w.strip(DELIMITERS) for w in text]
    for w in realtext:
        if w != '':
            WORDS.add(w)

# ascribe unique id num to each word
COUNT = 0
for word in WORDS:
    LOOKUP[word] = COUNT
    COUNT += 1

# increment occurrence counts per position in sparse matrix
for i in range(len(TWENTY_TRAIN.data)):
    datum = TWENTY_TRAIN.data[i]
    text = ' '.join(datum.split('\n')).split(' ')
    realtext = [w.strip(DELIMITERS) for w in text]
    for w in realtext:
        if w != '':
            POSITIONS[(i, LOOKUP[w])] += 1

# translate dict sparse matrix to data types for csr_matrix
for k, v in POSITIONS.items():
    ROW.append(k[0])
    COL.append(k[1])
    DATA.append(v)


SHAPE_BAG = (len(TWENTY_TRAIN.target), len(WORDS))
M_BAG = csr_matrix((array(DATA), (array(ROW), array(COL))), shape=SHAPE_BAG)
NORMS_BAG = get_row_norms(M_BAG)
AVG_SCORES_BAG = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]
Q_GROUP_AVG_BAG = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]


print('Tokenizing training data...')
M_TOKEN = COUNT_VECT.fit_transform(TWENTY_TRAIN.data)
NORMS_TOKEN = get_row_norms(M_TOKEN)
AVG_SCORES_TOKEN = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]
Q_GROUP_AVG_TOKEN = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]


print('Calculating tfidf values...')
M_TFIDF = TFIDF_TRANSFORMER.fit_transform(M_TOKEN)
NORMS_TFIDF = get_row_norms(M_TFIDF)
AVG_SCORES_TFIDF = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]
Q_GROUP_AVG_TFIDF = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]

# 2.3
GROUPS = [partition[:100] for partition in DOCUMENT_PARTITION]
QUERIES = [partition[100:len(partition)] for partition in DOCUMENT_PARTITION]
THRESHOLDS = [i/10 for i in range(11)]
PRECISIONS_BAG, RECALLS_BAG = [[0 for i in range(11)] for x in range(2)]
PRECISIONS_TOKEN, RECALLS_TOKEN = [[0 for i in range(11)] for x in range(2)]
PRECISIONS_TFIDF, RECALLS_TFIDF = [[0 for i in range(11)] for x in range(2)]


# 2.3.2
for t in range(len(THRESHOLDS)-1):
    count = 0
    tol = THRESHOLDS[t]
    for newsgroup_id in range(0, len(QUERIES)):
        NR = len(DOCUMENT_PARTITION[newsgroup_id])
        for query_idx in QUERIES[newsgroup_id]:
            # given a query and threshold, calculate average precision/recall over all newsgroups
            if random.randrange(0, ONE_OUT_OF) != 0:
                continue
            dr_bag, dt_bag, dr_token, dt_token, dr_tfidf, dt_tfidf = [0 for x in range(6)]
            for group_id in range(0, len(GROUPS)):
                for doc_idx in GROUPS[group_id]:
                    # bag of words
                    query_vector = M_BAG.getrow(query_idx).toarray()[0]
                    doc_vector = M_BAG.getrow(doc_idx).toarray()[0]
                    if cos_sim(query_vector, query_idx, doc_vector, doc_idx, NORMS_BAG) > tol:
                        dt_bag += 1
                        if group_id == newsgroup_id:
                            dr_bag += 1
                    # tokenized
                    query_vector = M_TOKEN.getrow(query_idx).toarray()[0]
                    doc_vector = M_TOKEN.getrow(doc_idx).toarray()[0]
                    if cos_sim(query_vector, query_idx, doc_vector, doc_idx, NORMS_TOKEN) > tol:
                        dt_token += 1
                        if group_id == newsgroup_id:
                            dr_token += 1
                    # tfidf
                    query_vector = M_TFIDF.getrow(query_idx).toarray()[0]
                    doc_vector = M_TFIDF.getrow(doc_idx).toarray()[0]
                    if cos_sim(query_vector, query_idx, doc_vector, doc_idx, NORMS_TFIDF) > tol:
                        dt_tfidf += 1
                        if group_id == newsgroup_id:
                            dr_tfidf += 1
            if dt_bag != 0:
                count += 1
                PRECISIONS_BAG[t] += dr_bag/dt_bag
            RECALLS_BAG[t] += dr_bag/NR
            if dt_token != 0:
                count += 1
                PRECISIONS_TOKEN[t] += dr_token/dt_token
            RECALLS_TOKEN[t] += dr_token/NR
            if dt_tfidf != 0:
                count += 1
                PRECISIONS_TFIDF[t] += dr_tfidf/dt_tfidf
            RECALLS_TFIDF[t] += dr_tfidf/NR
            print('count:', count, '| query:', query_idx, \
                    '| newsgroup:', TWENTY_TRAIN.target_names[newsgroup_id], '| threshold:', tol)
    if count != 0:
        PRECISIONS_BAG[t] /= count
        RECALLS_BAG[t] /= count
        PRECISIONS_TOKEN[t] /= count
        RECALLS_TOKEN[t] /= count
        PRECISIONS_TFIDF[t] /= count
        RECALLS_TFIDF[t] /= count
    print('bag of words: precisions', PRECISIONS_BAG)
    print('bag of words: recalls', RECALLS_BAG)
    print('tokenized: precisions', PRECISIONS_TOKEN)
    print('tokenized: recalls', RECALLS_TOKEN)
    print('tfidf: precisions', PRECISIONS_TFIDF)
    print('tfidf: recalls', RECALLS_TFIDF)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot(RECALLS_BAG, PRECISIONS_BAG)
plt.title('Precision-Recall Curve: Bag of Words')
plt.show()

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot(RECALLS_TOKEN, PRECISIONS_TOKEN)
plt.title('Precision-Recall Curve: Tokenized')
plt.show()

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot(RECALLS_TFIDF, PRECISIONS_TFIDF)
plt.title('Precision-Recall Curve: TFIDF')
plt.show()

"""
# 2.3.1
for a in range(N_NEWSGROUPS):
    for b in range(N_NEWSGROUPS):
        newsgrp_a = TWENTY_TRAIN.target_names[a]
        newsgrp_b = TWENTY_TRAIN.target_names[b]
        iters = math.floor((len(GROUPS[a])*len(QUERIES[b]))/ONE_OUT_OF)
        print('------------------------------------')
        print('newsgroups:', newsgrp_a, newsgrp_b)
        print('number of pairs:', iters)

        cos_sum_bag = 0
        cos_sum_token = 0
        cos_sum_tfidf = 0
        for doc_idx_a in GROUPS[a]:
            for doc_idx_b in QUERIES[b]:
                # sample
                if random.randrange(0, ONE_OUT_OF) != 0:
                    continue
                # bag of words
                v_a = M_BAG.getrow(doc_idx_a).toarray()[0]
                v_b = M_BAG.getrow(doc_idx_b).toarray()[0]
                cos_sum_bag += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_BAG)
                # tokenized
                v_a = M_TOKEN.getrow(doc_idx_a).toarray()[0]
                v_b = M_TOKEN.getrow(doc_idx_b).toarray()[0]
                cos_sum_token += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_TOKEN)
                # tfidf
                v_a = M_TFIDF.getrow(doc_idx_a).toarray()[0]
                v_b = M_TFIDF.getrow(doc_idx_b).toarray()[0]
                cos_sum_tfidf += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_TFIDF)

        Q_GROUP_AVG_BAG[a][b] = cos_sum_bag/iters
        print('avg sim on bag rep:\t', cos_sum_bag/iters)
        Q_GROUP_AVG_TOKEN[a][b] = cos_sum_token/iters
        print('avg sim on token rep:\t', cos_sum_token/iters)
        Q_GROUP_AVG_TFIDF[a][b] = cos_sum_tfidf/iters
        print('avg sim on tfidf rep:\t', cos_sum_tfidf/iters)

# show matrices as heatmaps
plt.imshow(Q_GROUP_AVG_BAG, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(Q_GROUP_AVG_TOKEN, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(Q_GROUP_AVG_TFIDF, cmap='hot', interpolation='nearest')
plt.show()
"""

""" 2.2
for a in range(N_NEWSGROUPS):
    for b in range(N_NEWSGROUPS):
        newsgrp_a = TWENTY_TRAIN.target_names[a]
        newsgrp_b = TWENTY_TRAIN.target_names[b]
        iters = math.floor((len(DOCUMENT_PARTITION[a])*len(DOCUMENT_PARTITION[b]))/ONE_OUT_OF)
        print('------------------------------------')
        print('newsgroups:', newsgrp_a, newsgrp_b)
        print('number of pairs:', iters)

        if AVG_SCORES_TOKEN[a][b] != -1:
            continue

        cos_sum_bag = 0
        cos_sum_token = 0
        cos_sum_tfidf = 0
        for doc_idx_a in DOCUMENT_PARTITION[a]:
            for doc_idx_b in DOCUMENT_PARTITION[b]:
                # sample 10%
                if random.randrange(0, ONE_OUT_OF) != 0:
                    continue
                # bag of words
                v_a = M_BAG.getrow(doc_idx_a).toarray()[0]
                v_b = M_BAG.getrow(doc_idx_b).toarray()[0]
                cos_sum_bag += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_BAG)
                # tokenized
                v_a = M_TOKEN.getrow(doc_idx_a).toarray()[0]
                v_b = M_TOKEN.getrow(doc_idx_b).toarray()[0]
                cos_sum_token += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_TOKEN)
                # tfidf
                v_a = M_TFIDF.getrow(doc_idx_a).toarray()[0]
                v_b = M_TFIDF.getrow(doc_idx_b).toarray()[0]
                cos_sum_tfidf += cos_sim(v_a, doc_idx_a, v_b, doc_idx_b, NORMS_TFIDF)

        AVG_SCORES_BAG[a][b] = cos_sum_bag/iters
        AVG_SCORES_BAG[b][a] = cos_sum_bag/iters
        print('avg sim on bag rep:\t', cos_sum_bag/iters)
        AVG_SCORES_TOKEN[a][b] = cos_sum_token/iters
        AVG_SCORES_TOKEN[b][a] = cos_sum_token/iters
        print('avg sim on token rep:\t', cos_sum_token/iters)
        AVG_SCORES_TFIDF[a][b] = cos_sum_tfidf/iters
        AVG_SCORES_TFIDF[b][a] = cos_sum_tfidf/iters
        print('avg sim on tfidf rep:\t', cos_sum_tfidf/iters)

# show matrices as heatmaps
plt.imshow(AVG_SCORES_BAG, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(AVG_SCORES_TOKEN, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(AVG_SCORES_TFIDF, cmap='hot', interpolation='nearest')
plt.show()
"""
