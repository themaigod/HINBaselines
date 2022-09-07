import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split

save_prefix = 'data/preprocessed/IMDB_processed/'
num_ntypes = 3

# load raw data, delete movies with no actor or director
movies = pd.read_csv('data/raw/IMDB/movie_metadata.csv', encoding='utf-8').dropna(
    axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)

# In[4]:


# extract labels, and delete movies with unwanted genres
# 0 for action, 1 for comedy, 2 for drama, -1 for others
labels = np.zeros((len(movies)), dtype=int)
for movie_idx, genres in movies['genres'].iteritems():
    labels[movie_idx] = -1
    for genre in genres.split('|'):
        if genre == 'Action':
            labels[movie_idx] = 0
            break
        elif genre == 'Comedy':
            labels[movie_idx] = 1
            break
        elif genre == 'Drama':
            labels[movie_idx] = 2
            break
unwanted_idx = np.where(labels == -1)[0]
movies = movies.drop(unwanted_idx).reset_index(drop=True)
labels = np.delete(labels, unwanted_idx, 0)

# In[5]:


# get director list and actor list
directors = list(set(movies['director_name'].dropna()))
directors.sort()
actors = list(set(movies['actor_1_name'].dropna().to_list() +
                  movies['actor_2_name'].dropna().to_list() +
                  movies['actor_3_name'].dropna().to_list()))
actors.sort()

# In[6]:


# build the adjacency matrix for the graph consisting of movies, directors and actors
# 0 for movies, 1 for directors, 2 for actors
dim = len(movies) + len(directors) + len(actors)
type_mask = np.zeros((dim), dtype=int)
type_mask[len(movies):len(movies) + len(directors)] = 1
type_mask[len(movies) + len(directors):] = 2

adjM = np.zeros((dim, dim), dtype=int)
for movie_idx, row in movies.iterrows():
    if row['director_name'] in directors:
        director_idx = directors.index(row['director_name'])
        adjM[movie_idx, len(movies) + director_idx] = 1
        adjM[len(movies) + director_idx, movie_idx] = 1
    if row['actor_1_name'] in actors:
        actor_idx = actors.index(row['actor_1_name'])
        adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
        adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
    if row['actor_2_name'] in actors:
        actor_idx = actors.index(row['actor_2_name'])
        adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
        adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
    if row['actor_3_name'] in actors:
        actor_idx = actors.index(row['actor_3_name'])
        adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
        adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1

# In[ ]:


# extract bag-of-word representations of plot keywords for each movie
# X is a sparse matrix
vectorizer = CountVectorizer(min_df=2)
movie_X = vectorizer.fit_transform(movies['plot_keywords'].fillna('').values)
# assign features to directors and actors as the means of their associated movies' features
adjM_da2m = adjM[len(movies):, :len(movies)]
adjM_da2m_normalized = scipy.sparse.csr_matrix(np.diag(1 / adjM_da2m.sum(axis=1))).dot(
    scipy.sparse.csr_matrix(adjM_da2m))
director_actor_X = scipy.sparse.csr_matrix(adjM_da2m_normalized).dot(movie_X)
full_X = scipy.sparse.vstack([movie_X, director_actor_X])

expected_metapaths = [
    [(0, 1, 0), (0, 2, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1)],
    [(2, 0, 2), (2, 0, 1, 0, 2)]
]
# create the directories if they do not exist
for i in range(num_ntypes):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
for i in range(num_ntypes):
    # get metapath based neighbor pairs
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
    # construct and save metapath-based networks
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)

    # save data
    # networkx graph (metapath specific)
    for G, metapath in zip(G_list, expected_metapaths[i]):
        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
    # node indices of edge metapaths
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

# save data
# all nodes adjacency matrix
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
# all nodes (movies, directors and actors) features
for i in range(num_ntypes):
    scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(i), full_X[np.where(type_mask == i)[0]])
# all nodes (movies, directors and actors) type labels
np.save(save_prefix + 'node_types.npy', type_mask)
# movie genre labels
np.save(save_prefix + 'labels.npy', labels)
# movie train/validation/test splits
rand_seed = 1566911444
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=3478, random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx=val_idx,
         train_idx=train_idx,
         test_idx=test_idx)
