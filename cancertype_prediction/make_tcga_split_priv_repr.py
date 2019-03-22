'''
Representation learning testing for TCGA data.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
import datetime
#import argparse

from common import expInterp, ensure_dir_exists
import batch

##########################################################################################
# SETUP
##########################################################################################
#set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

# input datasets
aux_data_set = "TCGA_split_pub_geneexpr"
data_set = "TCGA_split_priv_geneexpr"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'

# size of the representation to be learned
#repr_dims = [1, 2, 4, 8, 16]
repr_dims = [2, 4, 8, 12, 16]

normalize_data = False

algorithms = []

seeds = [0, 1, 2, 3, 4]
#seeds = [0, 1, 2]
#seeds = [3, 4]

#id_suffix = "-ess" # early stopping secondary (loss)
id_suffix = ""

# random projections
for seed in seeds:
  alg_id = "rand_proj"
  algorithms.append((
      alg_id,
      seed,
      lambda repr_dim, alg_id=alg_id, seed=seed:
      __import__('models.rand_proj').rand_proj.RandomProjection().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
  ))

# PCA
for seed in seeds:
  alg_id = "pca"
  algorithms.append((
      alg_id,
      seed,
      lambda repr_dim, alg_id=alg_id, seed=seed:
      __import__('models.pca').pca.PCA().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
  ))


# VAE
for prediction_var in ['gs', 'gi', 'ps', 'pi']:
  for seed in seeds:
    alg_id = "vae_torch_%s_e32_d32" % (prediction_var)
    algorithms.append((
        alg_id,
        seed,
        lambda repr_dim, alg_id=alg_id, seed=seed:
          __import__('models.vae_pytorch').vae_pytorch.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
      ))

for seed in seeds:
  alg_id = "vae_torch_%s_similar" % ('pi')
  algorithms.append((
      alg_id,
      seed,
      lambda repr_dim, alg_id=alg_id, seed=seed:
        __import__('models.vae_pytorch').vae_pytorch.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
    ))


# VAE
for prediction_var in ['gs', 'gi']:
  for seed in seeds:
    alg_id = "vae_%s_e32_d32_minmax01" % (prediction_var)
    algorithms.append((
        alg_id,
        seed,
        lambda repr_dim, alg_id=alg_id, seed=seed:
          __import__('models.vae_pytorch').vae_pytorch.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
      ))
    alg_id = "vae_%s_e32_d32_meanstd01" % (prediction_var)
    algorithms.append((
        alg_id,
        seed,
        lambda repr_dim, alg_id=alg_id, seed=seed:
          __import__('models.vae_pytorch').vae_pytorch.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
      ))
    alg_id = "vae_%s_e32_d32_5quantile01" % (prediction_var)
    algorithms.append((
        alg_id,
        seed,
        lambda repr_dim, alg_id=alg_id, seed=seed:
          __import__('models.vae_pytorch').vae_pytorch.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
      ))


##########################################################################################
# END OF SETUP
##########################################################################################

def mean_squared_error(x, x_pred):
  return np.average((x - x_pred) ** 2)

def relative_mean_squared_error(x, x_pred):
  mse = mean_squared_error(x, x_pred)
  x_avg = np.average(x, axis=0)
  return mse / np.average((x - x_avg) ** 2)

# the task function that is run with each argument combination
def task(args):
  repr_dim, (alg_id, seed, load_model) = args
  logging.info("representation size = %d, algorithm = %s, seed = %d", repr_dim, alg_id, seed)

  # read the PADS gene expression data
  logging.info("Reading gene expression data...")
  import pandas
  data = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  x = data.as_matrix()
  logging.info(" * data shape: %d x %d" % x.shape)
  
  #logging.info("Filter and normalize...")
  ## load gene names that appear also in TCGA data
  #tcga_gene_names = np.array(getHDF5data("data/%s_genes.h5" % (aux_data_set),
  #                                       True, False)[0], dtype=str)
  #in_tcga = np.array([(gene_name in tcga_gene_names) for gene_name in gene_names])
  #assert(np.sum(in_tcga) == len(tcga_gene_names))
  ## use only those genes
  #x = x[:,in_tcga]


  # normalize the input to _total_ unit variance and zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
    x -= np.mean(x, axis=0)

  # FIXME!
  #x = (x - np.amin(x,axis=0)) / (np.amax(x,axis=0) - np.amin(x,axis=0))
  #if alg_id != "pca":
  #x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
  
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  # init rng  
  np.random.seed(seed)
  
  # load the model
  logging.info("Loading the model...")
  alg = load_model(repr_dim)
  
  # get the representation
  logging.info("Computing the representation or size %d..." % (repr_dim))
  x_repr = alg.encode(x)

  # test to predict the data itself
  x_pred = alg.decode(x_repr)
  rel_mse = relative_mean_squared_error(x, x_pred)
  logging.info(" * reconstruct the data: rel_mse = %g", rel_mse)
  ensure_dir_exists("res")
  with open("res/private-encdec-rel_mse-%d-%s-%s-s%d%s.txt" %
            (repr_dim, aux_data_set, alg_id, seed, id_suffix),
            'w', encoding='utf-8') as f:
    f.write("%.6f\n" % rel_mse)
  
  # save the representation
  logging.info("Saving the representation...")
  ensure_dir_exists("data_repr")
  np.savetxt("data_repr/repr-%s-%d-%s-%s-s%d%s.csv" %
             (data_set, repr_dim, aux_data_set, alg_id, seed, id_suffix),
             x_repr, delimiter=',')


########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(repr_dims, algorithms))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()

