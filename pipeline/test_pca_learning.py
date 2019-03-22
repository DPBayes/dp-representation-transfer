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

# data types
data_types = [
  'linear',
  'sparselinear',
  'onelayer1',
  'twolayers1',
  'threelayers1',
]

# seeds
seeds = [0]

# ids of algorithms to run
algs_to_run = 'all'
#algs_to_run = ['ae_linear_pretrainpca']
#algs_to_run = ['ae_linear_initpca']

# the dimension of the representation to be learned
repr_dim = 5

n_epochs = 200

algorithms = []

# PCA
#alg = (
#  "pca",
#  "PCA (principal component analysis)",
#  lambda input_dim, repr_dim:
#    __import__('models.pca').pca.PCA().init(
#      input_dim = input_dim,
#      output_dim = repr_dim)
#)
#algorithms.append(alg)


for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
  alg = (
    "e_" + optimizer,
    "Linear autoencoder",
    lambda input_dim, repr_dim, optimizer=optimizer:
      __import__('models.encoder').encoder.Encoder().init(
        input_dim = input_dim,
        enc_dims = [],
        output_dim = repr_dim,
        enc_activations = 'linear',
        batch_normalization=False,
        init_pca = False,
        n_epochs = n_epochs,
        batch_size = 100,
        optimizer=optimizer)
  )
  algorithms.append(alg)

for lr in [0.1, 0.01, 0.001]:
  alg = (
    "e_SGD_%g" % lr,
    "Linear autoencoder",
    lambda input_dim, repr_dim, lr=lr:
      __import__('models.encoder').encoder.Encoder().init(
        input_dim = input_dim,
        enc_dims = [],
        output_dim = repr_dim,
        enc_activations = 'linear',
        batch_normalization=False,
        init_pca = False,
        n_epochs = n_epochs,
        batch_size = 100,
        optimizer=__import__('keras.optimizers').optimizers.SGD(lr=lr))
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

  x_test = x_train

  logging.info("Running and evaluating the algorithm...")
  
  # init the algorithm
  alg = makeAlg(data_dim, repr_dim)
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  from sklearn.decomposition import PCA as sk_PCA
  pca = sk_PCA(n_components=repr_dim)
  pca.fit(x_train)
  y_train = pca.transform(x_train)
  y_test = pca.transform(x_test)

  # define the progress saving function
  progress_filename = 'res/progress-enc-mse-%s-%d-%s.txt' % (data_type, seed, algName)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  def save_progress():
    y_test_pred = alg.encode(x_test)
    rel_mse = relative_mean_squared_error(y_test, y_test_pred)
    progress_file.write("%g\n" % rel_mse)
  
  # fit
  alg.learn(x_train, y_train,
            log_file_prefix=("log/%s-%d-%s" % (data_type, seed, algName)),
            callbacks=[save_progress])




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
