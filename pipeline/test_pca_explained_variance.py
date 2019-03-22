"""
Runs PCA on a given dataset and outputs the ratio of explained variance for each component.
"""
import numpy as np
from sklearn.decomposition import PCA as sk_PCA
import logging

from common import ensure_dir_exists
import batch

##########################################################################################
# SETUP
##########################################################################################
#set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

# input dataset
data_set = "TCGA_geneexpr_filtered_redistributed"
data_type = 'redistributed_gene_expressions'
#data_set = "TCGA_geneexpr_filtered"
#data_type = 'rnaseq_tpm_rle_log2_gene_expressions'

normalize_data = True

##########################################################################################
# END OF SETUP
##########################################################################################

def task(args):
  import pandas
  #data_set, = args
  logging.info("dataset = %s", data_set)
  # read the data sets
  logging.info("Reading data...")
  data = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  logging.info(" * gene expression shape: %d x %d" % data.shape)

  x = data.as_matrix()

  if normalize_data:
    # these shouldn't affect the results
    x -= np.mean(x)
    x /= np.std(x)
    x -= np.mean(x, axis=0)

  logging.info("Running PCA...")
  pca = sk_PCA()
  pca.fit(x)

  logging.info("Writing results...")
  res_dir = 'res/pca-explained-variance'
  res_filename = "%s/%s.txt" % (res_dir, data_set)
  ensure_dir_exists(res_dir)
  np.savetxt(res_filename, pca.explained_variance_ratio_)
  #pca.explained_variance_


# init and run
batch.init(task=task, args_ranges=())
batch.main()
