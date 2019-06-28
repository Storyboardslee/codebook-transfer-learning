"""
Generate toy matrix with cluster signals in both rows and columns with desired row and column cluster number and cluster size.
"""
import random
from random import gauss
from random import randint
import numpy as np

def generate_cluster_mat(cluster_size):
  cluster_num = randint(1,6)
  cluster_mat = [[gauss(cluster_num,0.2) for s in range(cluster_size)] for s in range(cluster_size) ]
  return cluster_mat

def concatent_cluster_mat(k,l,cluster_size):
  #concatenate rows
  rows = [np.concatenate([generate_cluster_mat(cluster_size) for i in range(l)],axis = 1) for i in range(k)]
  return np.concatenate(rows,axis = 0)

def shuffle(X):
  #randomly shuffle rows and columns of the matrix
  X_copy = np.copy(X)
  row_idx = list(range(X.shape[0]))
  column_idx = list(range(X.shape[1]))
  random.shuffle(row_idx)
  random.shuffle(column_idx)
  X_copy = X_copy[row_idx,:]
  X_copy = X_copy[:,column_idx]
  return X_copy

def generate_toy_data(k,l,cluster_size,symmetrize,shuffle):
    toy_data = concatent_cluster_mat(k,l,cluster_size)
    if symmetrize:
        assert k == l
        toy_data = np.fmax(toy_data,toy_data.T)
    if shuffle:
        toy_data = shuffle(toy_data)
    return toy_data

# Randomly hold out data and see if this inteferes with constructing codebook
# can't do X_test = np.nan, only 0

def train_test_split(X,hidden_fraction,symmetric):
  M,N = X.shape
  measured_idx = np.array([(i,j) for i,j in zip(*np.where(~np.isnan(X)))])
  n_known = len(measured_idx)
  n_hidden = int(hidden_fraction * n_known)
  shuffle = np.random.permutation(n_known)
  
  if symmetric:
    upper_measured_idx = np.array([(i,j) for i,j in measured_idx if i>=j])
    train_upper_idx = upper_measured_idx[shuffle[n_hidden:]]
    train_idx = [(i,j) for i,j in train_upper_idx] + [(j,i) for i,j in train_upper_idx]
  
  if not symmetric:
    train_idx = list(map(tuple,measured_idx[shuffle[n_hidden:]]))
  
  test_idx = list(set(list(map(tuple,measured_idx)))- set(train_idx))
  
  I_train = np.zeros((M,N),dtype = bool)
  I_train[tuple(np.array(train_idx).T)] = True
  I_test = np.zeros((M,N),dtype = bool)
  I_test[tuple(np.array(test_idx).T)] = True
  

  X_train = np.copy(X)
  X_test = np.copy(X)
  X_train[~I_train] = np.nan
  X_test[~I_test] = np.nan
  
  
  return X_train, X_test