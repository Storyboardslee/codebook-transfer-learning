# codebook-transfer-learning (COTL)
Transfer learning using ONMTF and codebook construction to extract information with more than one latent factors from source domain matrix and transfer information to sparse target domain. Based and modified from Li et al 2009, and Ding et al 2006. Implemented for cross-domain biological data imputation. Currently experiments are run with toy data.

## Setup
### Install dependencies required to run experiments
It is recommended that users install dependencies of COTL, into a Python environment using [Conda](https://conda.io/miniconda.html). To install Python and other dependencies, which you can do directly using the provided `environment.yml file`:

    conda env create -f environment.yml
    source activate codebook-transfer learning

### Repeat experiments with toy data
To recover missing entries in sparse target toy matrix with desired number of row and column clusters as well as hidden fraction, use the following command:
    
    python src/repeat_experiments_toy_data.py -k [# of row clusters] -l [#of column clusters] -hf [hidden fraction]
 
