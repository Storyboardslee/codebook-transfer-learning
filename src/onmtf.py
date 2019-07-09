"""
Implemented based on section 5 of 
H. Q. Ding, Chris & Li, Tao & Peng, Wei & Park, Haesun. (2006). 
Orthogonal nonnegative matrix t-factorizations for clustering. 
Proceedings of the ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining. 
2006. 126-135. 10.1145/1150402.1150420. 
"""
import numpy as np, numpy.linalg as LA
from numpy.linalg import multi_dot
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

class ONMTF(object):
  def __init__(self, X, k, l, max_iter):
    """
    X = source matrix
    k = rank
    l = column cluster
    """
    self.X = X
    self.m, self.n = X.shape
    self.k = k 
    self.l = l
    self.max_iter = max_iter
    
  @staticmethod
  def cluster(X,k):
    kmeans = KMeans(n_clusters=k,random_state=0).fit(X)
    labels = set(kmeans.labels_)
    labeled_features = kmeans.labels_
    return np.array([np.multiply([i==k for i in labeled_features],1) for k in labels]).T.astype(np.float64)
    
  
  def check_non_negativity(self):
    if self.X.min()<0:
      raise ValueError('X contains negative value')
      
    
  def frobenius_norm(self):
    """
    ||X-FSGt||^2, euclidean error between X and WH
    """
    self.FSGt = multi_dot([self.F,self.S,self.G.T])
    return LA.norm (self.X - self.FSGt)
  
  def initialize(self):
    """
    initialize F, G, and S latent matrices
    According to Ding et al. 2006:
    1. G is obtained via k means clustering of columns of X. G = G+0.2
    2. F is obtained via k means clustering of rows of X. F = F+ 0.2
    3. S is obtained via S = F.T@X@G 
    """
    # Initialization based on Ding et al 2006 Sec.5
    self.G = self.cluster(self.X.T,self.l)
    self.G += 0.2
    
    self.F = self.cluster(self.X, self.k)
    self.F += 0.2
    
    self.S = multi_dot([self.F.T,self.X,self.G])
    
  def initialize_random(self):
    # Random non-negative matrix initialization of F, S, and G
    self.F = np.random.rand(self.m, self.k).astype(np.float64)
    self.G = np.random.rand(self.n, self.l).astype(np.float64)
    self.S = np.random.rand(self.k, self.l).astype(np.float64)
 
  def update_F(self):
    enum = multi_dot([self.X,self.G,self.S.T])
    denom = multi_dot([self.F,self.F.T,self.X,self.G,self.S.T])
    self.F *= enum
    self.F /= denom
  
  def update_G(self):
    enum = multi_dot([self.X.T,self.F,self.S])
    denom = multi_dot([self.G,self.G.T,self.X.T,self.F,self.S])
    self.G *= enum
    self.G /= denom
  
  def update_S(self):
    enum = multi_dot([self.F.T, self.X,self.G])
    denom = multi_dot([self.F.T,self.F,self.S,self.G.T,self.G])
    self.S *= enum
    self.S /= denom
    
  def factorize(self,initialize_opt='kmeans'):
    # default initialization method is kmeans
    # check to see if X is non negative
    self.check_non_negativity()
    # initialize if no F,S,G as attribute
    if not hasattr(self,'F') or not hasattr(self,'G') or not hasattr(self,'S'):
      if initialize_opt == 'random':
        self.initialize_random()
      elif initialize_opt == 'kmeans':
        self.initialize()
      
    # generate a 1*max_iter array to record frob error of each iteration  
    self.frob_error_log = np.zeros(self.max_iter)
    
    # generate a 1*mzx_iter array to record r2 score of each iteration
    self.r2_scores = np.zeros(self.max_iter)
    
    # iteratively update F,S,G
    # does order matter much? 
    for i in range(self.max_iter):
      self.update_G()
      self.update_F()
      self.update_S()
      
      self.frob_error_log[i] = self.frobenius_norm()
      self.r2_scores[i] = r2_score(self.X, self.FSGt)
      
      if i%10 ==0:
        print('iteration: {} \nfrobenius error:{} \nR2 score:{} \n'.format(i, self.frob_error_log[i],self.r2_scores[i]))
      
      #early stop
      if i >= 1*self.max_iter and self.frob_error_log[i] >= self.frob_error_log[i-1]:
        print('Stopping early due to non-decreasing loss')
        break
      
      if i>= 1*self.max_iter and self.r2_scores[i] <= self.r2_scores[i-1]:
        print('Stopping early due to non-increasing R2 metric')
        break
      
