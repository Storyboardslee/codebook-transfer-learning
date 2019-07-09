# Load modules
from onmtf import *
from plot import *
from codebook_construction import *
from codebook_transfer import *
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def run_single_experiment(source_data, k,l,target_data, validation_data):
  #ONMTF
  source = ONMTF(source_data,k,l,max_iter=1000)
  source.factorize()
    
  #Codebook construction
  B = codebook_construction(source.X,source.F,source.G,binarize=False)
    
  #codebook_transfer
  transfer = CB_TRANSFER(target_data,B,max_iter=1000)
  transfer.search_local_minimum()
  transfer.fill_matrix()
  
  prediction = transfer.predict[~transfer.mask]
  test = validation_data[~transfer.mask]
  
  r2 = r2_score(prediction, test)
  r = pearsonr(prediction, test)
  
  return r, r2, B, transfer

def repeat_experiments(source_data,k,l,target_data,validation_data,repeat):

    r2scores = []
    rscores = []
    codebooks = []
    imputations = []
    for i in range(repeat):
        print('EXPERIMENT ITERATION: {}'.format(i))
        r, r2, codebook, imputation = run_single_experiment(source_data,k,l,target_data, validation_data)
        rscores.append(r)
        r2scores.append(r2)
        codebooks.append(codebook)
        imputations.append(imputation)
    
    return r2scores, rscores, codebooks, imputations
