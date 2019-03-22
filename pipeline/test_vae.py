'''
VAE testing.
'''

import numpy as np
import logging
#import argparse

from common import expInterp, ensure_dir_exists
import batch

##########################################################################################
# SETUP
##########################################################################################
#set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

repr_dim = 1

# seeds
#seeds = [0, 1, 2, 3, 4, 5]
seeds = [0]

algs_to_run = 'all'

n_epochs = 200

save_pred = True

algorithms = []

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
#algorithms.append(alg)

# VAE
alg = (
  "vae",
  "Variational autoencoder",
  lambda input_dim, repr_dim:
    __import__('models.vae').vae.VAE().init(
      input_dim = input_dim,
      latent_dim = repr_dim,
      enc_dims = [],
      #enc_dims = [2 * input_dim],
      dec_dims = 'same',
      enc_activations = 'relu',
      dec_activations = 'relu',
      prediction_var='global_same',
      optimizer = 'Adam',
      optimizer_params = {'lr': 0.001},
      n_epochs = n_epochs,
      batch_size = 1)
)
#algorithms.append(alg)

# VAE
alg = (
  "vae_torch",
  "Variational autoencoder",
  lambda input_dim, repr_dim:
    __import__('models.vae_pytorch').vae_pytorch.VAE().init(
      input_dim = input_dim,
      latent_dim = repr_dim,
      enc_dims = [],
      #enc_dims = [2 * input_dim],
      dec_dims = 'same',
      enc_activations = 'relu',
      dec_activations = 'relu',
      prediction_var = 'global_same',
      #prediction_var = np.array([0.0]),
      optimizer = 'Adam',
      optimizer_params = {'lr': 0.001},
      n_epochs = n_epochs,
      batch_size = 100)
)
algorithms.append(alg)

if algs_to_run != "all":
  algorithms = [a for a in algorithms if (a[0] in algs_to_run)]

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
  seed, (algName, _, makeAlg) = args
  data_type = "vae_test"
  logging.info("datatype = %s, seed = %d, algorithm = %s", data_type, seed, algName)
  np.random.seed(seed)
  x = np.random.normal(0.0, 1.0, (1000, 2))
  x = np.dot(x, np.array([[5.0, 3.0],[0.3, -0.5]]))
  data_dim = x.shape[1]
  logging.info(" * training set: %d x %d" % x.shape)
  logging.info(" * testing set: %d x %d" % x.shape)
  # init rng  

  logging.info("Running and evaluating the algorithm...")
  logging.info(" * using representation with dimension = %d", repr_dim)
  
  # init the algorithm
  alg = makeAlg(data_dim, repr_dim)
  
  # create output dir if does not exist
  #ensure_dir_exists('res')

  # define the progress saving function
  #progress_filename = 'res/progress-encdec-mse-%s-%d-%s.txt' % (data_type, seed, algName)
  #progress_file = open(progress_filename, 'w', encoding='utf-8')
  #def save_progress():
  #  x_test_pred = alg.decode(alg.encode(x_test))
  #  rel_mse = relative_mean_squared_error(x_test, x_test_pred)
  #  progress_file.write("%g\n" % rel_mse)

  # fit to the training data
  alg.learn(x,
            log_file_prefix=("log/%s-%d-%s" % (data_type, seed, algName)))
  
  x_test = x

  # test with the testing data
  x_test_pred = alg.decode(alg.encode(x_test))
  #x_test_pred = alg.decode_generate(alg.encode(x_test))
  #x_test_pred = alg.decode_generate(alg.encode_generate(x_test))
  #x_test_pred = alg.decode(alg.encode_generate(x_test))
  ensure_dir_exists('pred')
  data_filename = 'data/generated/%s-%d' % (data_type, seed)
  pred_filename = 'pred/final-encdec-%s-%d-%s' % (data_type, seed, algName)
  if save_pred:
    np.save(data_filename, x_test)
    np.save(pred_filename, x_test_pred)
  #from sklearn import metrics
  #mse = metrics.mean_squared_error(x_test, x_test_pred,
  #    multioutput='uniform_average')
  #explained_var = metrics.explained_variance_score(x_test, x_test_pred,
  #    multioutput='uniform_average')
  mse = mean_squared_error(x_test, x_test_pred)
  rel_mse = relative_mean_squared_error(x_test, x_test_pred)

  logging.info("Result: rel_mse = %g", rel_mse)



########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(seeds, algorithms))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()
