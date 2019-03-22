'''
Representation learning testing.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
#import argparse

from common import expInterp, ensure_dir_exists
import dataReader
import batch

#optional imports for data visualization
#import matplotlib.pyplot as plt
#import pylab

##########################################################################################
# SETUP
##########################################################################################
#set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

## data types
#data_types = [
#  'parabola',
#  'u',
#  'circle',
#]
## the dimension of the representation to be learned
#repr_dim = 1

## data types
#data_types = [
#  'sphere',
#]
## the dimension of the representation to be learned
#repr_dim = 2

# data types
data_types = [
  'linear',
#  'supersparse',
  'sparselinear',
  'onelayer1',
  'twolayers1',
  'threelayers1',
]
# the dimension of the representation to be learned
repr_dim = 5

# seeds
seeds = [0]

# ids of algorithms to run
algs_to_run = [
  'pca',
  'ae_linear',
#  'ae_linear_initpca',
#  'ae_linear_pretrainpca',
  'ae1',
#  'ae2',
#  'ae2_dropout',
#  'ae2b_dropout',
#  'ae2c_dropout',
#  'ae2_pretrainpca',
#  'ae3',
#  'vae',
]
algs_to_run = 'all'

n_epochs = 100

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = True

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
  "ae_linear",
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
      batch_size = 100,
      optimizer='Adam')
)
algorithms.append(alg)


# One layer AE
for optimizer in ['Adam']:
  alg = (
    "ae1_" + optimizer,
    "One layer autoencoder",
    lambda input_dim, repr_dim, optimizer=optimizer:
      __import__('models.ae').ae.AE().init(
        input_dim = input_dim,
        enc_dims = [],
        output_dim = repr_dim,
        dec_dims = "same",
        #enc_activations = 'relu',
        #enc_activations = 'prelu',
        enc_activations = 'elu',
        dec_activations = 'linear',
        n_epochs = n_epochs,
        batch_size = 100,
        batch_normalization=False,
        optimizer=optimizer)
  )
  algorithms.append(alg)

# Two layer AE
for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
  alg = (
    "ae2_" + optimizer,
    "Two layer autoencoder",
    lambda input_dim, repr_dim, optimizer=optimizer:
      __import__('models.ae').ae.AE().init(
        input_dim = input_dim,
        #enc_dims = [np.rint(expInterp(input_dim, repr_dim, 0.5))],
        enc_dims = [8*input_dim],
        output_dim = repr_dim,
        dec_dims = "same",
        enc_activations = ['prelu', 'prelu'],
        dec_activations = ['prelu', 'linear'],
        n_epochs = n_epochs,
        batch_size = 100,
        optimizer=optimizer)
  )
  algorithms.append(alg)

# Two layer AE
alg = (
  "ae2_dropout",
  "Two layer autoencoder with dropout",
  lambda input_dim, repr_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [np.rint(expInterp(input_dim, repr_dim, 0.5))],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['softplus', 'softplus'],
      dec_activations = ['softplus', 'tanh'],
      dropout = 0.2,
      n_epochs = n_epochs,
      batch_size = 100)
)
#algorithms.append(alg)

# Two layer AE
alg = (
  "ae2b_dropout",
  "Two layer autoencoder with dropout",
  lambda input_dim, repr_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [input_dim],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['softplus', 'softplus'],
      dec_activations = ['softplus', 'tanh'],
      dropout = 0.2,
      n_epochs = n_epochs,
      batch_size = 100)
)
#algorithms.append(alg)

# Two layer AE
alg = (
  "ae2c_dropout",
  "Two layer autoencoder with dropout",
  lambda input_dim, repr_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [input_dim * 3],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['softplus', 'softplus'],
      dec_activations = ['softplus', 'tanh'],
      dropout = 0.2,
      n_epochs = n_epochs,
      batch_size = 100)
)
#algorithms.append(alg)

# Two layer AE with pca pre training
alg = (
  "ae2_pretrainpca",
  "Two layer autoencoder with PCA pre training",
  lambda input_dim, repr_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [np.rint(expInterp(input_dim, repr_dim, 0.5))],
      output_dim = repr_dim,
      dec_dims = "same",
      enc_activations = ['softplus', 'softplus'],
      dec_activations = ['softplus', 'tanh'],
      pre_train = ('pca', (n_epochs,)),
      n_epochs = n_epochs,
      batch_size = 100)
)
#algorithms.append(alg)

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
  data_type, seed, (algName, _, makeAlg) = args
  logging.info("datatype = %s, seed = %d, algorithm = %s", data_type, seed, algName)
  # read the data sets
  logging.info("Reading data...")
  y_train, x_train, y_test, x_test = dataReader.main("%s_%d" % (data_type, seed))
  data_dim = x_train.shape[1]
  logging.info(" * training set: %d x %d" % x_train.shape)
  logging.info(" * testing set: %d x %d" % x_test.shape)
  # init rng  
  np.random.seed(seed)

  logging.info("Running and evaluating the algorithm...")
  logging.info(" * using representation with dimension = %d", repr_dim)
  
  # init the algorithm
  alg = makeAlg(data_dim, repr_dim)
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  # define the progress saving function
  progress_filename = 'res/progress-encdec-mse-%s-%d-%s.txt' % (data_type, seed, algName)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  def save_progress():
    x_test_pred = alg.decode(alg.encode(x_test))
    rel_mse = relative_mean_squared_error(x_test, x_test_pred)
    progress_file.write("%g\n" % rel_mse)

  # fit to the training data
  alg.learn(x_train,
            log_file_prefix=("log/%s-%d-%s" % (data_type, seed, algName)),
            callbacks=[save_progress])
  
  # TODO: remove?
  x_test = x_train

  # test with the testing data
  x_test_pred = alg.decode(alg.encode(x_test))
  ensure_dir_exists('pred')
  pred_filename = 'pred/final-encdec-%s-%d-%s' % (data_type, seed, algName)
  if save_pred:
    np.save(pred_filename, x_test_pred)
  #from sklearn import metrics
  #mse = metrics.mean_squared_error(x_test, x_test_pred,
  #    multioutput='uniform_average')
  #explained_var = metrics.explained_variance_score(x_test, x_test_pred,
  #    multioutput='uniform_average')
  mse = mean_squared_error(x_test, x_test_pred)
  rel_mse = relative_mean_squared_error(x_test, x_test_pred)

  logging.info("Result: rel_mse = %g", rel_mse)
  logging.info("Writing results to a file...")
  res_filename = 'res/final-encdec-mse-%s-%d-%s.txt' % (data_type, seed, algName)
  with open(res_filename, 'w', encoding='utf-8') as f:
    f.write("data = %-16s seed = %-4d alg = %-10s " % (data_type, seed, algName))
    f.write("mse = %.6f  " % mse)
    f.write("rel_mse = %.6f  " % rel_mse)
    f.write("\n")



########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(data_types, seeds, algorithms))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()
