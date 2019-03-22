'''
Representation learning testing for TCGA data.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
import datetime
#import argparse

from common import expInterp, ensure_dir_exists
import dataReader
import batch

from readHDF5 import getHDF5data

#optional imports for data visualization
#import matplotlib.pyplot as plt
#import pylab

##########################################################################################
# SETUP
##########################################################################################
#set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

# input datasets
aux_data_set = "TCGA_geneexpr_filtered_redistributed"
data_set = "GDSC_geneexpr_filtered"
data_type = 'rma_gene_expressions'

#aux_data_set = "TCGA_geneexpr_filtered"
#data_set = "GDSC_geneexpr_filtered_redistributed"
#data_type = 'redistributed_gene_expressions'

# size of the representation to be learned
repr_dims = [10]

normalize_data = True

algorithms = []

seeds = [0, 1, 2]#, 3, 4]

id_suffix = ""
#id_suffix = "-ess" # early stopping secondary (loss)

# PCA
'''algorithms.append((
    'pca',
    0, #seed
    lambda repr_dim, alg_id='pca':
      __import__('models.pca').pca.PCA().load("repr_models/%s-%d-%s" %
                                          (aux_data_set, repr_dim, alg_id))
  ))'''

# autoencoders
'''optimizer = 'Adam'
for alg in [
                'ae1aux_64xR_dropout12',
                'ae2aux_32xR_dropout12',
                'ae3aux_32xR_dropout12',
              ]:
  for weight in [0.0, .001, .002, .003, .005, .010, .015, .025, .05, .15, .5, (1-.15), (1-.05), (1-.015), (1-.005), 1.0]:
    for seed in seeds:
      alg_id = "%s_w%s_%s" % (alg, weight, optimizer)
      algorithms.append((
          alg_id,
          seed,
          lambda repr_dim, alg_id=alg_id, seed=seed:
            __import__('models.ae').ae.AE().load("repr_models/%s-%d-%s-s%d%s" %
                                                (aux_data_set, repr_dim, alg_id, seed, id_suffix))
        ))'''


'''for seed in seeds:
  algorithms.append((
      'ae0_linear',
      seed, #seed
      lambda repr_dim, alg_id='ae0_linear':
        __import__('models.ae').ae.AE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
    ))'''


# AE
'''optimizer = 'Adam'
for alg in [
                'ae1_64xR_dropout12',
                'ae2_32xR_dropout12',
                'ae3_32xR_dropout12',
              ]:

  for seed in seeds:
    alg_id = "%s_%s" % (alg, optimizer)
    algorithms.append((
        alg_id,
        seed,
        lambda repr_dim, alg_id=alg_id, seed=seed:
          __import__('models.ae').ae.AE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
      ))'''


# VAE
optimizer = 'Adam'
for prediction_var in ['gs', 'gi', 'ps', 'pi']:
  for alg in [
                'vae_%s_e64' % (prediction_var),
                'vae_%s_e32e4' % (prediction_var),
              ]:

    for seed in seeds:
      alg_id = "%s_%s" % (alg, optimizer)
      algorithms.append((
          alg_id,
          seed,
          lambda repr_dim, alg_id=alg_id, seed=seed:
            __import__('models.vae').vae.VAE().load("repr_models/%s-%d-%s-s%d%s" % (aux_data_set, repr_dim, alg_id, seed, id_suffix))
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
  np.savetxt("data_repr/repr-%d-%s-%s-s%d%s.csv" %
             (repr_dim, aux_data_set, alg_id, seed, id_suffix),
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

