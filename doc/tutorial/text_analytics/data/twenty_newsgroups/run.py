#!/ilab/users/sfj19/anaconda3/bin/python
""" Script to test my stuff """

# imports
import random
from collections import defaultdict
from numpy import dot, array, add
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

# Lemma:
# [sum_i][sum_j] dot(v_i, w_j)/(norm(v_i)*norm(w_j)
# = [sum_i] dot( v_i/norm(v_i), [sum_i] w_j/norm(w_j) )
def marginal_unit_vector_sums(partitions, models):
    """
    Input:
        - groups: list of sublists, where each sublist contains doc vectors representing newsgroups
        - queries: list of sublists, where each sublist contains doc vectors representing query docs
        - models: list of models where each model has a different data vector representation
    Output:
        - summations: dict of lists indexed first by data vector rep and second by newsgroup
            each element is a vector yielded through summation over all unit vectors of a newsgroup
    """
    print('Calculating marginal newsgroup unit vector summations...')
    n_partitions = len(partitions)
    summations = defaultdict()
    for data_rep in DATA_REPS:
        summations[data_rep] = [0 for a in range(len(partitions))]

    for data_rep, model in models.items():
        for n_id in range(n_partitions):
            for doc_index in partitions[n_id]:
                vector = model.getrow(doc_index).toarray()[0]
                summations[data_rep][n_id] = add(summations[data_rep][n_id], vector/norm(vector))

    return summations


# constants
TWENTY_TRAIN = fetch_20newsgroups()
COUNT_VECT = CountVectorizer()
TFIDF_TRANSFORMER = TfidfTransformer()
N_NEWSGROUPS = len(TWENTY_TRAIN.target_names)

# set up matrices, norms, and final scores for each data representation method
print('Building model: BAG')
WORDS = set()
DELIMITERS = '/\\\'\"!@#$%^&*()-+{}[]:;.,~`\t\n<>?=_'

ROW, COL, DATA = [[] for l in range(3)]
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

M_BAG = csr_matrix((array(DATA), (array(ROW), array(COL))),
                   shape=(len(TWENTY_TRAIN.target), len(WORDS)))

print('Building model: TOKEN...')
M_TOKEN = COUNT_VECT.fit_transform(TWENTY_TRAIN.data)

print('Building model: TFIDF...')
M_TFIDF = TFIDF_TRANSFORMER.fit_transform(M_TOKEN)

# partition newsgroups
DOCUMENT_PARTITION = [[] for x in range(N_NEWSGROUPS)]
for i in range(len(TWENTY_TRAIN.target)):
    t = TWENTY_TRAIN.target[i]
    DOCUMENT_PARTITION[t].append(i)

DATA_REPS = ['BAG', 'TOKEN', 'TFIDF']
MODELS = {'BAG': M_BAG, 'TOKEN': M_TOKEN, 'TFIDF': M_TFIDF}
GROUPS = [partition[:100] for partition in DOCUMENT_PARTITION]
QUERIES = [partition[100:len(partition)] for partition in DOCUMENT_PARTITION]
THRESHOLDS = [i/10 for i in range(11)]
SCORE_MAPS = defaultdict()
NORMS = defaultdict()


for rep in DATA_REPS:
    NORMS[rep] = get_row_norms(MODELS[rep])
    SCORE_MAPS[rep] = [[-1 for x in range(N_NEWSGROUPS)] for x in range(N_NEWSGROUPS)]

print('Doing the thing!')


# """
print('Problem 2.2...')
DOC_SUMS = marginal_unit_vector_sums(DOCUMENT_PARTITION, MODELS)

for a in range(N_NEWSGROUPS):
    for b in range(N_NEWSGROUPS):
        iters = len(DOCUMENT_PARTITION[a]*len(DOCUMENT_PARTITION[b]))
        print('---\navg sims for newsgroups: {0} {1} for {2} total pairs'.format(
            TWENTY_TRAIN.target_names[a], TWENTY_TRAIN.target_names[b], iters))
        if SCORE_MAPS[DATA_REPS[0]][a][b] != -1:
            continue
        for rep, score_map in SCORE_MAPS.items():
            score_map[a][b] = dot(DOC_SUMS[rep][a], DOC_SUMS[rep][b])/iters
            print('{0}\t{1}'.format(rep, score_map[a][b]))

# show matrices as heatmaps
for rep, score_map in SCORE_MAPS.items():
    plt.title('Average Similarities b/w Newsgroups: {}'.format(rep))
    plt.imshow(score_map, cmap='hot', interpolation='nearest')
    plt.show()
# """


# """
print('Problem 2.3.2...')
GROUP_SUMS = marginal_unit_vector_sums(GROUPS, MODELS)
QUERY_SUMS = marginal_unit_vector_sums(QUERIES, MODELS)

for a in range(N_NEWSGROUPS):
    for b in range(N_NEWSGROUPS):
        iters = len(GROUPS[a])*len(QUERIES[b])
        print('---\navg sims for newsgroups: {0} {1} for {2} total pairs'.format(
            TWENTY_TRAIN.target_names[a], TWENTY_TRAIN.target_names[b], iters))
        for rep, score_map in SCORE_MAPS.items():
            score_map[a][b] = dot(GROUP_SUMS[rep][a], QUERY_SUMS[rep][b])/iters
            print('{0}\t{1}'.format(rep, score_map[a][b]))

# show matrices as heatmaps
for rep, score_map in SCORE_MAPS.items():
    plt.title('Average Similarities b/w Queries/Newsgroups: {}'.format(rep))
    plt.imshow(score_map, cmap='hot', interpolation='nearest')
    plt.show()
# """

# """ Feel free to tinker with the value of ONE_OUT_OF - this is a sampling rate
print('Problem 2.3.2...')
ONE_OUT_OF = 1000
PRECISIONS, RECALLS, COUNTS, DR, DT = [defaultdict() for x in range(5)]
for rep in DATA_REPS:
    RECALLS[rep], PRECISIONS[rep] = [[0 for i in range(len(THRESHOLDS))] for j in range(2)]

for t in range(len(THRESHOLDS)-1):
    for rep in DATA_REPS:
        COUNTS[rep] = 0
        # given a query, threshold, and data rep, calculate avg precision/recall over all newsgroups
        for newsgroup_id in range(N_NEWSGROUPS):
            for query_idx in QUERIES[newsgroup_id]:
                if random.randrange(0, ONE_OUT_OF) != 0:
                    continue
                DR[rep], DT[rep] = [0 for j in range(2)]
                for group_id in range(N_NEWSGROUPS):
                    for doc_idx in GROUPS[group_id]:
                        query = MODELS[rep].getrow(query_idx).toarray()[0]
                        doc = MODELS[rep].getrow(doc_idx).toarray()[0]
                        if cos_sim(query, query_idx, doc, doc_idx, NORMS[rep]) > THRESHOLDS[t]:
                            DT[rep] += 1
                            if group_id == newsgroup_id:
                                DR[rep] += 1
                if DT[rep] != 0:
                    COUNTS[rep] += 1
                    PRECISIONS[rep][t] += DR[rep]/DT[rep]
                RECALLS[rep][t] += DR[rep]/len(GROUPS[newsgroup_id])
                print('{0}\trep: {1}\tquery: {2}\tgroup: {3}\tthreshold: {4}'.format(COUNTS[rep],\
                        rep, query_idx, TWENTY_TRAIN.target_names[newsgroup_id], THRESHOLDS[t]))
        if COUNTS[rep] != 0:
            PRECISIONS[rep][t] /= COUNTS[rep]
            RECALLS[rep][t] /= COUNTS[rep]
        print('{0}\tprecisions: {1}\n{3}\trecalls: {2}'.\
                format(rep, PRECISIONS[rep], rep, RECALLS[rep]))

for rep in DATA_REPS:
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(RECALLS[rep], PRECISIONS[rep])
    plt.title('Precision-Recall Curve: {}'.format(rep))
    plt.show()
# """
scikit-learn/doc/tutorial/text_analytics/data/twenty_newsgroups/
