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

# data types
data_sets = [
  'geneMatrix',
]

# the dimension of input and the representation to be learned
#input_dim = None
#input_dim = 2000
#repr_dims = [10, 20]
#repr_dims = [10, 50, 100, 200, 500, 1000]
#input_dim = 500
#repr_dims = [10, 20]
#repr_dims = [10, 20, 50, 100, 200]
#input_dim = 200
#repr_dims = [10, 20]
#repr_dims = [10, 20, 50, 100]
#input_dim = 50
#repr_dims = [10, 20]
#repr_dims = [5, 10, 20, 30]

#input_dims = [50]
#input_dims = [50, 200, 500]
#input_dims = [2000, 5000, None]
#input_dims = [50, 200, 500, 2000, 5000]
input_dims = [66, 200, 2000, None]
#input_dims = [66, 200]
repr_dims = [10]

include_target = True


algs_to_run = 'all'

n_epochs = 2000
deadline = None
#max_duration = None
#max_duration = datetime.timedelta(hours=1)
max_duration = datetime.timedelta(hours=14)

batch_size = 100

normalize_data = True
#validation_split = 0
validation_split = 0.2

# logging settings (what to log)
log_weights = False

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = False

algorithms = []

# Linear AE
alg = (
  "ae0_linear_Adam",
  "Linear autoencoder",
  lambda input_dim, repr_dim, reconstruction_dim:
    __import__('models.ae').ae.AE().init(
      input_dim = input_dim,
      enc_dims = [],
      output_dim = repr_dim,
      reconstruction_dim = reconstruction_dim,
      dec_dims = "same",
      enc_activations = 'linear',
      dec_activations = 'linear',
      n_epochs = n_epochs,
      batch_size = batch_size,
      optimizer='Adam',
      log_weights=log_weights)
)
algorithms.append(alg)

# Two layer AE
#if False:
for optimizer in ['Adam']:
  for hidden_dim_mult in [16, 32, 64]:
  #for hidden_dim_mult in [8]:
    alg = (
      "ae1_%sxR_%s" % (hidden_dim_mult, optimizer),
      "Two layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, hidden_dim_mult=hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [hidden_dim_mult*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'linear'],
          dec_activations = ['prelu', 'linear'],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae1_%sxR_dropout25_%s"  % (hidden_dim_mult, optimizer),
      "Two layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, hidden_dim_mult=hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [hidden_dim_mult*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'linear'],
          dec_activations = ['prelu', 'linear'],
          dropout=[0.2, 0.5],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae1_%sxR_dropout12_%s"  % (hidden_dim_mult, optimizer),
      "Two layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, hidden_dim_mult=hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [hidden_dim_mult*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'linear'],
          dec_activations = ['prelu', 'linear'],
          dropout=[0.1, 0.2],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae1_%sxR_batchnorm_%s"  % (hidden_dim_mult, optimizer),
      "Two layer autoencoder with batch normalizaton",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, hidden_dim_mult=hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [hidden_dim_mult*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'linear'],
          dec_activations = ['prelu', 'linear'],
          batch_normalization=True,
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)


# Three layer AE
#if False:
for optimizer in ['Adam']:
  for first_hidden_dim_mult in [16, 32, 64]:
  #for first_hidden_dim_mult in [8]:
    alg = (
      "ae2_%sxR_%s" % (first_hidden_dim_mult, optimizer),
      "Three layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'linear'],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae2_%sxR_dropout25_%s"  % (first_hidden_dim_mult, optimizer),
      "Three layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'linear'],
          dropout=[0.2, 0.5, 0.5],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    #algorithms.append(alg)
    alg = (
      "ae2_%sxR_dropout12_%s"  % (first_hidden_dim_mult, optimizer),
      "Three layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'linear'],
          dropout=[0.1, 0.2, 0.2],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae2_%sxR_batchnorm_%s"  % (first_hidden_dim_mult, optimizer),
      "Three layer autoencoder with batch normalizaton",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'linear'],
          batch_normalization=True,
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)

# Four layer AE
#if False:
for optimizer in ['Adam']:
  for first_hidden_dim_mult in [16, 32, 64]:
  #for first_hidden_dim_mult in [8]:
    alg = (
      "ae3_%sxR_%s" % (first_hidden_dim_mult, optimizer),
      "Four layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 8*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae3_%sxR_dropout25_%s"  % (first_hidden_dim_mult, optimizer),
      "Four layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 8*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dropout=[0.2, 0.5, 0.5, 0.5],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae3_%sxR_dropout12_%s"  % (first_hidden_dim_mult, optimizer),
      "Four layer autoencoder",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 8*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dropout=[0.1, 0.2, 0.2, 0.2],
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
    )
    algorithms.append(alg)
    alg = (
      "ae3_%sxR_batchnorm_%s"  % (first_hidden_dim_mult, optimizer),
      "Four layer autoencoder with batch normalizaton",
      lambda input_dim, repr_dim, reconstruction_dim,
             optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult:
        __import__('models.ae').ae.AE().init(
          input_dim = input_dim,
          enc_dims = [first_hidden_dim_mult*repr_dim, 8*repr_dim, 4*repr_dim],
          output_dim = repr_dim,
          reconstruction_dim = reconstruction_dim,
          dec_dims = "same",
          enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
          batch_normalization=True,
          n_epochs = n_epochs,
          batch_size = batch_size,
          optimizer=optimizer,
          log_weights=log_weights,
          log_weights_diff_norm=2)
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
  data_set, input_dim, repr_dim, (algName, _, makeAlg) = args
  logging.info("dataset = %s, algorithm = %s", data_set, algName)
  # read the data sets
  logging.info("Reading data...")
  x = getHDF5data("data/%s.h5" % (data_set), True, True)[0]
  # transpose and redo the log transform
  x = x.T
  x = np.log1p(x)
  logging.info(" * data shape: %d x %d" % x.shape)

  # load gene names
  features = getHDF5data("data/%s_genes.h5" % (data_set), True, False)[0]

  # load and find target genes
  target_features = np.genfromtxt('data/70genes.txt',dtype='S16')
  #targets = [i for i in range(len(features)) if features[i] in target_features]
  in_target = np.array([(features[i] in target_features) for i in range(len(features))])
  targets = np.where(in_target)[0]
  not_targets = np.where(~in_target)[0]

  # normalize the input to _total_ unit variance and zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
  
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
  
  # extract the target
  y = x[:, targets]
  val_y = val_x[:, targets]

  #in_target = np.zeros(x.shape[1], dtype=bool)
  #in_target[targets] = True
  #not_targets = np.arange(x.shape[1])[~in_target]
  #not_targets = np.where(in_target)

  # optionally reduce dimensionality
  if include_target:
    if input_dim is not None:
      additionals = np.random.choice(not_targets, input_dim - len(targets), replace=False)
      input_features = np.sort(np.concatenate((targets, additionals)))
      x = x[:, input_features]
      val_x = val_x[:, input_features]
  else:
    assert(False)

  # make data size a multiple of batch size (drop extra rows)
  #n = (x.shape[0] // batch_size) * batch_size
  #x = x[0:n, :]
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)
  logging.info(" * target shape: %d x %d" % y.shape)
  # init rng  
  np.random.seed(0)

  logging.info("Running and evaluating the algorithm...")
  logging.info(" * using representation with dimension = %d", repr_dim)
  
  # init the algorithm
  alg = makeAlg(data_dim, repr_dim, len(targets))
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  # define the progress saving function
  progress_filename = 'res/progress-encdec65-mse-%s-%s-%d-%s.txt' % (data_set, input_dim, repr_dim, algName)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  if val_x is not None:
    val_progress_filename = 'res/progress-encdec65-validation-mse-%s-%s-%d-%s.txt' % (data_set, input_dim, repr_dim, algName)
    val_progress_file = open(val_progress_filename, 'w', encoding='utf-8')
  def save_progress():
    y_pred = alg.decode(alg.encode(x))
    rel_mse = relative_mean_squared_error(y, y_pred)
    progress_file.write("%g\n" % rel_mse)
    if val_x is not None:
      val_y_pred = alg.decode(alg.encode(val_x))
      rel_mse = relative_mean_squared_error(val_y, val_y_pred)
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
  alg.learn(x, y=y,
            log_file_prefix=("log/%s-%s-%d-65-%s" % (data_set, input_dim, repr_dim, algName)),
            per_epoch_callback_funs=[save_progress], callbacks=callbacks)
  
  # test fitting
  y_pred = alg.decode(alg.encode(x))
  ensure_dir_exists('pred')
  pred_filename = 'pred/final-encdec65-%s-%s-%d-%s' % (data_set, input_dim, repr_dim, algName)
  if save_pred:
    np.save(pred_filename, x_pred)
  #from sklearn import metrics
  #mse = metrics.mean_squared_error(x, x_pred,
  #    multioutput='uniform_average')
  #explained_var = metrics.explained_variance_score(x, x_pred,
  #    multioutput='uniform_average')
  mse = mean_squared_error(y, y_pred)
  rel_mse = relative_mean_squared_error(y, y_pred)

  logging.info("Result: rel_mse = %g", rel_mse)
  logging.info("Writing results to a file...")
  res_filename = 'res/final-encdec65-mse-%s-%s-%d-%s.txt' % (data_set, input_dim, repr_dim, algName)
  with open(res_filename, 'w', encoding='utf-8') as f:
    f.write("data = %-16s input_dim = %-6s repr_dim = %-6d alg = %-10s " %
            (data_set, input_dim, repr_dim, algName))
    f.write("mse = %.6f  " % mse)
    f.write("rel_mse = %.6f  " % rel_mse)
    f.write("\n")



########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(data_sets, input_dims, repr_dims, algorithms))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()

