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

# input dataset
#data_set = "TCGA_geneexpr_filtered_redistributed"
#data_type = 'redistributed_gene_expressions'
data_set = "TCGA_geneexpr_filtered"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'

# size of the representation to be learned
repr_dims = [10]

algs_to_run = 'all'

n_epochs = 2000
deadline = None
#max_duration = None
#max_duration = datetime.timedelta(hours=1)
max_duration = datetime.timedelta(hours=4)

batch_size = 64

normalize_data = True
#validation_split = 0
validation_split = 0.2

early_stopping = True

# logging settings (what to log)
log_weights = False

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = False

algorithms = []

# PCA
alg = (
  "pca",
  "PCA (principal component analysis)",
  lambda input_dim, repr_dim:
    __import__('models.pca').pca.PCA().init(
      input_dim = input_dim,
      output_dim = repr_dim)
)
algorithms.append(alg)

# Linear AE
alg = (
  "ae0_linear",
  "Linear autoencoder",
  lambda input_dim, repr_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = 'linear',
      dec_activations = 'linear',
      n_epochs = n_epochs,
      batch_size = batch_size,
      optimizer='Adam',
      log_weights=log_weights)
)
algorithms.append(alg)


optimizer = 'Adam'

# one hidden layer
'''hidden_dim_mult = 64
alg = (
  "ae1_%sxR_dropout12_%s"  % (hidden_dim_mult, optimizer),
  "Two layer autoencoder",
  lambda input_dim, repr_dim,
          optimizer=optimizer, hidden_dim_mult=hidden_dim_mult:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [hidden_dim_mult*repr_dim],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['prelu', 'linear'],
      dec_activations = ['prelu', 'linear'],
      dropout=[0.1, 0.2],
      n_epochs = n_epochs,
      batch_size = batch_size,
      optimizer=optimizer,
      early_stopping=early_stopping,
      log_weights=log_weights,
      log_weights_diff_norm=2)
)
algorithms.append(alg)


# two hidden layers
first_hidden_dim_mult = 32
alg = (
  "ae2_%sxR_dropout12_%s"  % (first_hidden_dim_mult, optimizer),
  "Three layer autoencoder",
  lambda input_dim, repr_dim,
          optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['prelu', 'prelu', 'linear'],
      dec_activations = ['prelu', 'prelu', 'linear'],
      dropout=[0.1, 0.2, 0.2],
      n_epochs = n_epochs,
      batch_size = batch_size,
      optimizer=optimizer,
      early_stopping=early_stopping,
      log_weights=log_weights,
      log_weights_diff_norm=2)
)
algorithms.append(alg)

# three hidden layers
first_hidden_dim_mult = 32
alg = (
  "ae3_%sxR_dropout12_%s"  % (first_hidden_dim_mult, optimizer),
  "Four layer autoencoder",
  lambda input_dim, repr_dim,
          optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim, 4*repr_dim],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
      dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
      dropout=[0.1, 0.2, 0.2, 0.2],
      n_epochs = n_epochs,
      batch_size = batch_size,
      optimizer=optimizer,
      early_stopping=early_stopping,
      log_weights=log_weights,
      log_weights_diff_norm=2)
)
algorithms.append(alg)'''


# more layers
if False:
#for optimizer in ['Adam']:
  for n_hidden_layers in [3, 4, 5, 6, 7, 8]:
  #for n_hidden_layers in [5]:
    for max_hidden_dim_mult in [8, 16, 32]:
    #for max_hidden_dim_mult in [8]:
      common_params = dict(
        dec_dims = "same",
        enc_activations = n_hidden_layers * ['prelu'] + ['linear'],
        dec_activations = n_hidden_layers * ['prelu'] + ['linear'],
        n_epochs = n_epochs,
        batch_size = batch_size,
        optimizer = optimizer,
        log_weights = log_weights,
        log_weights_diff_norm = 2
      )
      enc_dim_mults = [max(max_hidden_dim_mult,2**a)
                       for a in range(n_hidden_layers,0,-1)]
      alg = (
        "ae%db_%sxR_dropout25_%s" % (n_hidden_layers, max_hidden_dim_mult, optimizer),
        "%d-layer autoencoder" % (n_hidden_layers),
        lambda input_dim, repr_dim,
               common_params=common_params, enc_dim_mults=enc_dim_mults:
          __import__('models.ae').ae.AE().init(
              **common_params,
              **dict(
                input_dim = input_dim,
                enc_dims = [a*repr_dim for a in enc_dim_mults],
                output_dim = repr_dim,
                dropout = [0.2] + n_hidden_layers * [0.5]
              )
            )
      )
      algorithms.append(alg)
      alg = (
        "ae%db_%sxR_dropout12_%s" % (n_hidden_layers, max_hidden_dim_mult, optimizer),
        "%d-layer autoencoder" % (n_hidden_layers),
        lambda input_dim, repr_dim,
               common_params=common_params, enc_dim_mults=enc_dim_mults:
          __import__('models.ae').ae.AE().init(
              **common_params,
              **dict(
                input_dim = input_dim,
                enc_dims = [a*repr_dim for a in enc_dim_mults],
                output_dim = repr_dim,
                dropout = [0.1] + n_hidden_layers * [0.2]
              )
            )
      )
      algorithms.append(alg)
      alg = (
        "ae%db_%sxR_batchnorm_%s" % (n_hidden_layers, max_hidden_dim_mult, optimizer),
        "%d-layer autoencoder" % (n_hidden_layers),
        lambda input_dim, repr_dim,
               common_params=common_params, enc_dim_mults=enc_dim_mults:
          __import__('models.ae').ae.AE().init(
              **common_params,
              **dict(
                input_dim = input_dim,
                enc_dims = [a*repr_dim for a in enc_dim_mults],
                output_dim = repr_dim,
                batch_normalization=True,
              )
            )
      )
      algorithms.append(alg)

if algs_to_run != "all":
  algorithms = [a for a in algorithms if (a[0] in algs_to_run)]

##########################################################################################
# END OF SETUP
##########################################################################################

def mean_squared_error(x, x_pred):
  #from sklearn import metrics
  #mse = metrics.mean_squared_error(x, x_pred,
  #    multioutput='uniform_average')
  #explained_var = metrics.explained_variance_score(x, x_pred,
  #    multioutput='uniform_average')
  #return np.average(np.average((x - x_pred) ** 2, axis=0))
  return np.average((x - x_pred) ** 2)

def relative_mean_squared_error(x, x_pred):
  mse = mean_squared_error(x, x_pred)
  x_avg = np.average(x, axis=0)
  #return mse / np.average(np.average((x - x_avg) ** 2, axis=0))
  return mse / np.average((x - x_avg) ** 2)

# the task function that is run with each argument combination
def task(args):
  import pandas
  repr_dim, (alg_id, _, make_alg) = args
  logging.info("dataset = %s, algorithm = %s", data_set, alg_id)
  # read the data sets
  logging.info("Reading data...")
  #x = getHDF5data("data/%s.h5" % (data_set), True, True)[0]
  ## transpose and redo the log transform
  #x = x.T
  #x = np.log1p(x)
  data = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  x = data.as_matrix()
  logging.info(" * data shape: %d x %d" % x.shape)

  #x = x[:,0:20] # FIXME: POISTA

  # normalize the input to _total_ unit variance and per-feature zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
    x -= np.mean(x, axis=0)

  # init rng  
  np.random.seed(0)
  
  # separate validation set if needed
  val_x = None
  if validation_split:
    logging.info("Splitting into training and validation sets")
    m = x.shape[0]
    perm = np.random.permutation(m)
    x = x[perm,:]
    split_point = int(validation_split * m)
    (val_x, x) = (x[:split_point,:], x[split_point:,:])
    logging.info(" * training set shape: %d x %d" % x.shape)
    logging.info(" * validation set shape: %d x %d" % val_x.shape)
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  logging.info("Running the algorithm...")
  logging.info(" * learning a representation of size %d", repr_dim)
  
  # init the algorithm
  alg = make_alg(data_dim, repr_dim)
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  # define the progress saving function
  progress_filename = 'res/progress-encdec-mse-%s-%d-%s.txt' % (data_set, repr_dim, alg_id)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  if val_x is not None:
    val_progress_filename = 'res/progress-encdec-validation-mse-%s-%d-%s.txt' % (data_set, repr_dim, alg_id)
    val_progress_file = open(val_progress_filename, 'w', encoding='utf-8')
  def save_progress():
    x_pred = alg.decode(alg.encode(x))
    rel_mse = relative_mean_squared_error(x, x_pred)
    progress_file.write("%g\n" % rel_mse)
    if val_x is not None:
      val_x_pred = alg.decode(alg.encode(val_x))
      rel_mse = relative_mean_squared_error(val_x, val_x_pred)
      val_progress_file.write("%g\n" % rel_mse)
  
  callbacks = []
  # add stopping callback
  if deadline is not None:
    from models.nncommon import TimeBasedStopping
    callbacks.append(TimeBasedStopping(deadline=deadline, verbose=1))
  elif max_duration is not None:
    from models.nncommon import TimeBasedStopping
    callbacks.append(TimeBasedStopping(max_duration=max_duration, verbose=1))

  # fit to the training data
  alg.learn(x, validation_data=val_x,
            log_file_prefix=("log/%s-%d-%s" % (data_set, repr_dim, alg_id)),
            per_epoch_callback_funs=[save_progress], callbacks=callbacks)
  
  # save model
  logging.info("Saving the learned model...")
  ensure_dir_exists('repr_models')
  alg.save("repr_models/%s-%d-%s" % (data_set, repr_dim, alg_id))


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

