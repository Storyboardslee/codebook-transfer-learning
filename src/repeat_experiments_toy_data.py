# Load modules
from onmtf import *
from plot import *
from codebook_construction import *
from codebook_transfer import *
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from toy_data_generation import *
import sys, os, argparse
import json
import numpy as np

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
  
  return r, r2, #B, transfer

def repeat_experiments(source_data,k,l,target_data,validation_data,repeat):

    #codebooks = []
    #imputations = []
    all_results = []
    for i in range(repeat):
        print('EXPERIMENT ITERATION: {}'.format(i))
        r, r2 = run_single_experiment(source_data,k,l,target_data, validation_data)
        print('Pearsonr: {}'.format(r))
        print('R2: {}'.format(r2))
        results = {'pearson_r': r[0],'pearsonr_pval': r[1],'r2':r2}
        #imputed_results = {'codebook':codebook,'imputations':imputation}
        #codebooks.append(codebook)
        #imputations.append(imputation)
        all_results.append(results)
    
    
    return all_results

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-k','--row_cluster', type = int, required=True)
  parser.add_argument('-l','--column_cluster', type = int, required=True)
  
  parser.add_argument('-c','--cluster_size', type = int, required=False, default=10 )
  
  parser.add_argument('-hf','--hidden_fraction', type = float, required=True)
  parser.add_argument('-shf','--shuffle', type = bool, required=False,default= False)
  
  parser.add_argument('-sym','--symmetrize', type = bool, required=False,default = False)
  
  parser.add_argument('-r','--repeat', type = int, required=False, default = 100)
  return parser

def collect_dict_items(dicts):
    collected = {}
    for k in dicts[0].keys():
        collected[k] = [d[k] for d in dicts]
    return collected

def summarize_results(result_dicts):
    collected = collect_dict_items(result_dicts)
    summarized = {}
    for k, vals in collected.items():
        summarized[k] = {'mean':np.mean(vals),
                         'std':np.std(vals),
                         'min':np.min(vals),
                         'max':np.max(vals),}
    return summarized, collected


def main():
  args = get_parser().parse_args(sys.argv[1:])
  k = args.row_cluster
  l = args.column_cluster
  hf = args.hidden_fraction
  c = args.cluster_size
  sym = args.symmetrize
  shf = args.shuffle

  source_data = generate_toy_data(k,l,c,sym,shf) 
  target_data, validation_data = train_test_split(source_data,hf,args.symmetric)
  all_results = repeat_experiments(source_data,k,l,target_data,validation_data,args.repeat) 
  #summarized, collected = summarize_results(all_results)

  with open('repeat_exp_toy_data_resuts.txt','w') as f:
    for result in all_results:
      f.write('{} \n'.format(result))


if __name__ == "__main__":
  main()
