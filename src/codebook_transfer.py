import numpy as np, numpy.linalg as LA
from numpy.linalg import multi_dot
from random import randint

class CB_TRANSFER():
  def __init__(self,Xtgt,B,max_iter):
    self.Xtgt = Xtgt
    self.B = B
    self.p, self.q = Xtgt.shape
    self.k, self.l = B.shape
    self.max_iter = max_iter
  
  @staticmethod
  def return_min_idx(a):
    #a is a list with numbers
    return min(range(len(a)), key=a.__getitem__)
  
  @staticmethod
  def non_j_idx(j,idx_range):
    non_j = list(range(idx_range))
    non_j.remove(j)
    return non_j
  
  def mask_nan(self):
    # save a copy of unmasked Xtgt
    self.Xtgt_unmasked = np.copy(self.Xtgt)
    # mask 
    self.Xtgt = np.nan_to_num(self.Xtgt)

  def generate_W(self):
    measured_idx = np.array([(i,j) for i,j in zip(*np.where(~np.isnan(self.Xtgt_unmasked)))])
    W = np.zeros((self.p,self.q))
    W_flipped = np.zeros((self.p,self.q))
    mask = np.zeros((self.p,self.q),dtype = bool)
    mask[tuple(measured_idx.T)] = True
    W[mask] = 1
    W_flipped[~mask] = 1
    self.mask = mask
    self.W = W
    self.W_flipped = W_flipped 
  
  def initialize_Vtgt(self):
    rows = []
    for i in range(self.q):
      row = np.zeros((1,self.l))
      j = randint(0,self.l-1)
      row[0,j] = 1
      rows.append(row)
    self.Vtgt = np.concatenate(rows,axis = 0)
  
  def initialize_Utgt(self):
    self.Utgt = np.zeros((self.p,self.k))
    
  def update_Utgt(self):
    BV_t = np.array(np.dot(self.B,self.Vtgt.T))
    for i in range(self.p):
      l2_norm_list = []
      diag_W = np.diag(self.W[i,:])
      for j in range(self.k):
        err_j = np.subtract(self.Xtgt[i,:],BV_t[j,:])
        l2_norm_j = err_j@diag_W@err_j.T
        l2_norm_list.append(l2_norm_j)
      
      j = np.nanargmin(l2_norm_list)
      self.Utgt[i,j] = 1
      self.Utgt[i,self.non_j_idx(j,self.k)] = 0
    
  def update_Vtgt(self):
    UB = np.array(np.dot(self.Utgt,self.B))
    for i in range(self.q): 
      l2_norm_list = []
      diag_W = np.diag(self.W[:,i])
      for j in range(self.l):
        err_j = np.array(self.Xtgt[:,i]-UB[:,j])
        l2_norm_j = err_j@diag_W@err_j.T
        l2_norm_list.append(l2_norm_j)
      
      j = np.nanargmin(l2_norm_list)
      self.Vtgt[i,j] = 1
      self.Vtgt[i,self.non_j_idx(j,self.l)] = 0
      
  def search_local_minimum(self):
    # mask Xtgt if nan exist in matrix
    if np.isnan(self.Xtgt).any():
      self.mask_nan()
    
    # generate W
    if not hasattr(self,'W') or not hasattr(self,'W_flipped'):
      self.generate_W()
      
    # initialize Utgt and Vtgt:
    if not hasattr(self,'Utgt'):
      self.initialize_Utgt()
    
    if not hasattr(self,'Vtgt'):
      self.initialize_Vtgt()
    
    # update Utgt and Vtgt
    for i in range(self.max_iter):
      self.update_Utgt()
      self.update_Vtgt()
      if i%10 ==0:
        print('iteration: {}'.format(i))
  
  def fill_matrix(self):
  
    self.predict = np.multiply(self.W_flipped,multi_dot([self.Utgt,self.B,self.Vtgt.T]))
    self.Xtgt_filled = np.multiply(self.W,self.Xtgt)+ self.predict
    

    