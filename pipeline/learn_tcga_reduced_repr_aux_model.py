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

# size of the representation to be learned
repr_dims = [10]

algs_to_run = 'all'

n_epochs = 2000
deadline = None
#max_duration = None
#max_duration = datetime.timedelta(minutes=1)
max_duration = datetime.timedelta(hours=4)

batch_size = 64

normalize_data = True
#validation_split = 0
validation_split = 0.2

early_stopping = True
#early_stopping = 'val_secondary_loss'

# logging settings (what to log)
log_weights = False
log_weights_diff_norm = None

# droputs (note: must be multiples of 0.1)
first_layer_dropout = 0.1; other_layer_dropouts = 0.2
#first_layer_dropout = 0.2; other_layer_dropouts = 0.4

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = False

# RNG seeds
seeds = [0, 1, 2]#, 3, 4]

id_suffix = ""
#id_suffix = "-ess" # early stopping secondary (loss)

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
#algorithms.append(alg)

# Linear AE
alg = (
  "ae0_linear_mlp",
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
#algorithms.append(alg)


optimizer = 'Adam'

for secondary_loss_weight in [0.0, .001, .002, .003, .005, .010, .015, .025, .05, .15, .5, (1-.15), (1-.05), (1-.015), (1-.005), 1.0]:
  # one hidden layer
  hidden_dim_mult = 64
  alg = (
    "ae1aux_%sxR_dropout%d%d_w%s_%s"  % (hidden_dim_mult, 10*first_layer_dropout, 10*other_layer_dropouts, secondary_loss_weight, optimizer),
    "Two layer autoencoder",
    lambda input_dim, repr_dim, num_classes,
            optimizer=optimizer, hidden_dim_mult=hidden_dim_mult,
            secondary_loss_weight=secondary_loss_weight:
      __import__('models.ae').ae.AE().init(
        input_dim = input_dim,
        enc_dims = [hidden_dim_mult*repr_dim],
        output_dim = repr_dim,
        dec_dims = "same",
        enc_activations = ['prelu', 'linear'],
        dec_activations = ['prelu', 'linear'],
        secondary_dims = [2*num_classes, num_classes],
        secondary_activations = ['prelu', 'softmax'],
        dropout=[first_layer_dropout, other_layer_dropouts],
        secondary_dropout=[other_layer_dropouts, other_layer_dropouts],
        secondary_loss = 'categorical_crossentropy',
        secondary_loss_weight = secondary_loss_weight,
        n_epochs = n_epochs,
        batch_size = batch_size,
        optimizer=optimizer,
        early_stopping=early_stopping,
        log_weights=log_weights,
        log_weights_diff_norm=log_weights_diff_norm)
  )
  algorithms.append(alg)


  # two hidden layers
  first_hidden_dim_mult = 32
  alg = (
    "ae2aux_%sxR_dropout%d%d_w%s_%s"  % (first_hidden_dim_mult, 10*first_layer_dropout, 10*other_layer_dropouts, secondary_loss_weight, optimizer),
    "Three layer autoencoder",
    lambda input_dim, repr_dim, num_classes,
            optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult,
            secondary_loss_weight=secondary_loss_weight:
      __import__('models.ae').ae.AE().init(
        input_dim = input_dim,
        enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim],
        output_dim = repr_dim,
        dec_dims = "same",
        enc_activations = ['prelu', 'prelu', 'linear'],
        dec_activations = ['prelu', 'prelu', 'linear'],
        secondary_dims = [2*num_classes, num_classes],
        secondary_activations = ['prelu', 'softmax'],
        dropout=[first_layer_dropout, other_layer_dropouts, other_layer_dropouts],
        secondary_dropout=[other_layer_dropouts, other_layer_dropouts],
        secondary_loss = 'categorical_crossentropy',
        secondary_loss_weight = secondary_loss_weight,
        n_epochs = n_epochs,
        batch_size = batch_size,
        optimizer=optimizer,
        early_stopping=early_stopping,
        log_weights=log_weights,
        log_weights_diff_norm=log_weights_diff_norm)
  )
  algorithms.append(alg)

  # three hidden layers
  first_hidden_dim_mult = 32
  alg = (
    "ae3aux_%sxR_dropout%d%d_w%s_%s"  % (first_hidden_dim_mult, 10*first_layer_dropout, 10*other_layer_dropouts, secondary_loss_weight, optimizer),
    "Four layer autoencoder",
    lambda input_dim, repr_dim, num_classes,
            optimizer=optimizer, first_hidden_dim_mult=first_hidden_dim_mult,
            secondary_loss_weight=secondary_loss_weight:
      __import__('models.ae').ae.AE().init(
        input_dim = input_dim,
        enc_dims = [first_hidden_dim_mult*repr_dim, 4*repr_dim, 4*repr_dim],
        output_dim = repr_dim,
        dec_dims = "same",
        enc_activations = ['prelu', 'prelu', 'prelu', 'linear'],
        dec_activations = ['prelu', 'prelu', 'prelu', 'linear'],
        secondary_dims = [2*num_classes, num_classes],
        secondary_activations = ['prelu', 'softmax'],
        dropout=[first_layer_dropout, other_layer_dropouts, other_layer_dropouts,
                 other_layer_dropouts],
        secondary_dropout=[other_layer_dropouts, other_layer_dropouts],
        secondary_loss = 'categorical_crossentropy',
        secondary_loss_weight = secondary_loss_weight,
        n_epochs = n_epochs,
        batch_size = batch_size,
        optimizer=optimizer,
        early_stopping=early_stopping,
        log_weights=log_weights,
        log_weights_diff_norm=log_weights_diff_norm)
  )
  algorithms.append(alg)


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
        log_weights_diff_norm = log_weights_diff_norm
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

def cross_entropy(x, x_pred):
  epsilon = 10e-8
  x_pred = np.clip(x_pred, epsilon, 1.0 - epsilon)
  return -np.average(np.sum(x * np.log(x_pred), axis=1))

def relative_cross_entropy(x, x_pred):
  x_avg = np.average(x, axis=0)
  return cross_entropy(x, x_pred) / cross_entropy(x, x_avg)

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
  repr_dim, (alg_id, _, make_alg), seed = args
  logging.info("dataset = %s, algorithm = %s", data_set, alg_id)
  # read the data sets
  logging.info("Reading data...")
  data = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  logging.info(" * gene expression shape: %d x %d" % data.shape)

  aux_target = pandas.read_hdf("data/TCGA_cancertype.h5", 'cancer_types')
  logging.info(" * auxiliary target size: %d" % aux_target.shape)

  common_samples = data.index.intersection(aux_target.index)
  data = data.loc[common_samples]
  aux_target = aux_target.loc[common_samples]
  logging.info(" * number of common samples: %d" % common_samples.size)

  from models.nncommon import categorical_to_binary

  x = data.as_matrix()
  y = categorical_to_binary(aux_target.values)
  num_classes = y.shape[1]

  # normalize the input to _total_ unit variance and per-feature zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
    x -= np.mean(x, axis=0)

  # init rng  
  np.random.seed(seed)
  
  # separate validation set if needed
  val_x = None
  val_y = None
  if validation_split:
    logging.info("Splitting into training and validation sets")
    m = x.shape[0]
    perm = np.random.permutation(m)
    x = x[perm,:]
    y = y[perm,:]
    split_point = int(validation_split * m)
    (val_x, x) = (x[:split_point,:], x[split_point:,:])
    (val_y, y) = (y[:split_point,:], y[split_point:,:])
    logging.info(" * training set shape: %d x %d" % x.shape)
    logging.info(" * validation set shape: %d x %d" % val_x.shape)
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  logging.info("Running the algorithm...")
  logging.info(" * learning a representation of size %d", repr_dim)
  
  # init the algorithm
  alg = make_alg(data_dim, repr_dim, num_classes)
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  full_model_id = "%s-%d-%s-s%d%s" % (data_set, repr_dim, alg_id, seed, id_suffix)

  # define the progress saving function
  progress_filename = 'res/progress-encdec-mse-%s.txt' % (full_model_id)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  aux_progress_filename = 'res/progress-aux-ce-%s.txt' % (full_model_id)
  aux_progress_file = open(aux_progress_filename, 'w', encoding='utf-8')
  if val_x is not None:
    val_progress_filename = 'res/progress-encdec-validation-mse-%s.txt' % (full_model_id)
    val_progress_file = open(val_progress_filename, 'w', encoding='utf-8')
    aux_val_progress_filename = 'res/progress-aux-validation-ce-%s.txt' % (full_model_id)
    aux_val_progress_file = open(aux_val_progress_filename, 'w', encoding='utf-8')
  def save_progress():
    x_pred = alg.decode(alg.encode(x))
    rel_mse = relative_mean_squared_error(x, x_pred)
    progress_file.write("%g\n" % rel_mse)
    aux_pred = alg.predict_secondary(x)
    aux_rel_ce = relative_cross_entropy(y, aux_pred)
    aux_progress_file.write("%g\n" % aux_rel_ce)
    if val_x is not None:
      val_x_pred = alg.decode(alg.encode(val_x))
      rel_mse = relative_mean_squared_error(val_x, val_x_pred)
      val_progress_file.write("%g\n" % rel_mse)
      val_aux_pred = alg.predict_secondary(val_x)
      aux_rel_ce = relative_cross_entropy(val_y, val_aux_pred)
      aux_val_progress_file.write("%g\n" % aux_rel_ce)
  
  callbacks = []
  # add stopping callback
  if deadline is not None:
    from models.nncommon import TimeBasedStopping
    callbacks.append(TimeBasedStopping(deadline=deadline, verbose=1))
  elif max_duration is not None:
    from models.nncommon import TimeBasedStopping
    callbacks.append(TimeBasedStopping(max_duration=max_duration, verbose=1))

  # fit to the training data
  alg.learn(x, secondary_y=y,
            validation_data=val_x, secondary_validation_data=val_y,
            log_file_prefix=("log/%s" % (full_model_id)),
            per_epoch_callback_funs=[save_progress], callbacks=callbacks)
  
  # save model
  logging.info("Saving the learned model...")
  ensure_dir_exists('repr_models')
  alg.save("repr_models/%s" % (full_model_id))


########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(repr_dims, algorithms, seeds))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()

