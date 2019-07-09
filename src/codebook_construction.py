"""
Implemented from Eq.(2) of 
Li, Bin & Yang, Qiang & Xue, Xiangyang. (2009). 
Can Movies and Books Collaborate? Cross-Domain Collaborative Filtering for Sparsity Reduction.. 
IJCAI International Joint Conference on Artificial Intelligence. 2052-2057. 
"""
import numpy as np
from numpy.linalg import multi_dot

def codebook_construction(X,F,G,binarize):
  """
  Codebook construction:
  1. Binarize F and G matrix with 0 and 1 (Li et al 2009 did this for simplicity, 
  but not sure how this step would change the resut)
  2. Compute codebook with the following equation
  B = [U.T@X@V]\varslash[U.T]
  """
  if binarize:
    F = binarize_matrix(F)
    G = binarize_matrix(G)
  #sum_vector_m = np.ones(X.shape[0])
  #sum_vector_n = np.ones(X.shape[1])
  sum_vector = np.ones((X.shape[0],X.shape[1]))
  dividend = multi_dot([F.T,X,G])
  divisor = multi_dot([F.T,sum_vector,G])
  
  # element-wise aka hadanard division to generate codebook B
  B = np.true_divide(dividend,divisor)
  
  return B

def binarize_matrix(F):
  """
  binarize input matrix with 0 and 1
  """
  F_copy = np.copy(F)
  for i in range(F_copy.shape[0]):
    #Find largest score in the row
    max_score = sorted(list(F_copy[i,:])).pop()
    #Replace all other scores into 0
    F_copy[i,:] = np.where(F_copy[i,:]==max_score,F_copy[i,:],0)
    #Repalce largest scoree with 1
    F_copy[i,:] = np.where(F_copy[i,:] ==0,F_copy[i,:],1)
  
  return F_copy
    
    