import numpy as np, numpy.linalg as LA
from numpy.linalg import multi_dot
from random import randint
"""
Implemented from Moreno et al 2012
"""

class CB_TRANSFER():
  def __init__(self,Xtgt,B,max_iter):
    # B is a list of codebooks
    self.Xtgt = Xtgt
    self.B = B
    self.p, self.q = Xtgt.shape
    self.k, self.l = B.shape
    self.max_iter = max_iter
    self.N = len(B)
  
  #@staticmethod
  #def return_min_idx(a):
    #a is a list with numbers
   # return min(range(len(a)), key=a.__getitem__)
  
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
    
  
  def initialize_Vtgt(self):
    rows = []
    for i in range(self.q):
      row = np.zeros((1,self.l))
      j = randint(0,self.l-1)
      row[0,j] = 1
      rows.append(row)
    Vn = np.concatenate(rows,axis = 0)
    self.Vtgt = dict()
    for n in range(self.N):
      self.Vtgt['V_{}'.format(n)] = Vn
  
  def initialize_Utgt(self):
    Un = np.zeros((self.p,self.k))
    self.Utgt = dict()
    for n in range(self.N):
      self.Utgt['U_{}'.format(n)] = Un
    
  def update_Utgt(self):
    for i in range(self.p):
      predict_sum = np.add([np.array(self.alpha['alpha_{}'.format(n)]*np.dot(self.B[n],self.Vtgt['V_{}'.format(n)])) for n in range(self.N)])
      err = [LA.norm(self.Xtgt[i,:]-predict_sum[j,:]) for j in range(self.k)]
      j = np.nanargmin(err)
    for n in range(self.N):
      self.Utgt[n][i,j] = 1
      self.Utgt[n][i,self.non_j_idx(j,self.k)] = 0
    
  def update_Vtgt(self,B,n):
    for i in range(self.q): 
      predict_sum = np.add([np.array(self.alpha['alpha_{}'.format(n)]*np.dot(self.Utgt,self.B[n])) for n in range(self.N)])
      err = [LA.norm(self.Xtgt[:,i]-predict_sum[:,j]) for j in range(self.l)]
      j = np.nanargmin(err)
      self.Vtgt[i,j] = 1
      self.Vtgt[i,self.non_j_idx(j,self.l)] = 0
  
  def initialize_alpha(self):
    #dict of alpha 
    self.alpha = dict()
    for i in range(self.N):
      self.alpha['alpha_{}'.format(i)] = 1/N
     
  def update_alpha(self):
    pass
    
      
  def search_local_minimum(self):
    # mask Xtgt if nan exist in matrix
    if np.isnan(self.Xtgt).any():
      self.mask_nan()
      
    # initialize Utgt and Vtgt:
    if not hasattr(self,'Utgt'):
      self.initialize_Utgt()
    
    if not hasattr(self,'Vtgt'):
      self.initialize_Vtgt()
    
    # update Utgt and Vtgt
    for i in range(self.max_iter):
      for n in range(self.N):
        self.update_Utgt(self.B[n])
        self.update_Vtgt(self.B[n])
        self.update_alpha()
      if i%10 ==0:
        print('iteration: {}'.format(i))
      
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
  
  def fill_matrix(self):
    if not hasattr(self,'W') or not hasattr(self,'W_flipped'):
      self.generate_W()
    
    self.predict = np.multiply(self.W_flipped,multi_dot([self.Utgt,self.B,self.Vtgt.T]))
    self.Xtgt_filled = np.multiply(self.W,self.Xtgt)+ self.predict
    
