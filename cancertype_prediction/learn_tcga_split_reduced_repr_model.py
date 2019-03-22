'''
Representation learning testing for TCGA data.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
import datetime
import time
import math

from common import expInterp, ensure_dir_exists, pretty_duration
import batch
from types import SimpleNamespace

######################################################################
# SETUP
######################################################################
# set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)

# input dataset
data_set = "TCGA_split_pub_geneexpr"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'

# size of the representation to be learned
repr_dims = [2, 4, 8, 12, 16]
#repr_dims = [16]

algs_to_run = 'all'
#algs_to_run = 'vae_gs_e32_d32_minmax01'

#n_epochs = 1
n_epochs = 2000
deadline = None
#max_duration = None
#max_duration = datetime.timedelta(minutes=5)
max_duration = datetime.timedelta(hours=1)

batch_size = 64

normalize_data = False
#validation_split = 0
validation_split = 0.2

early_stopping = True
#early_stopping = 'val_secondary_loss'

# logging settings (what to log)
log_weights = False

# droputs (note: must be multiples of 0.1)
#first_layer_dropout = 0.1; other_layer_dropouts = 0.2
#first_layer_dropout = 0.2; other_layer_dropouts = 0.4

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = False

# RNG seeds
seeds = [0, 1, 2, 3, 4]
#seeds = [0, 1, 2]
#seeds = [4]
#seeds = [0]

#id_suffix = "-ess" # early stopping secondary (loss)
id_suffix = ""

algorithms = []

# Random projection
alg = (
  "rand_proj",
  "Random projection (Gaussian)",
  lambda input_dim, repr_dim:
    __import__('models.rand_proj').rand_proj.RandomProjection().init(
      input_dim = input_dim,
      output_dim = repr_dim)
)
algorithms.append(alg)

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

for prediction_var in ['gs', 'gi', 'ps', 'pi']:
  alg = (
    "vae_torch_" + prediction_var + "_e32_d32",
    "Variational autoencoder",
    lambda input_dim, repr_dim, prediction_var=prediction_var:
      __import__('models.vae_pytorch').vae_pytorch.VAE().init(
        input_dim = input_dim,
        latent_dim = repr_dim,
        #enc_dims = [],
        enc_dims = [32*repr_dim],
        dec_dims = 'same',
        enc_activations = 'relu',
        dec_activations = 'relu',
        prediction_mean_activation = 'sigmoid',
        prediction_var = prediction_var,
        prediction_log_var_min = math.log(0.01**2),
        normalize_input_type = 'minmax',
        normalize_input_axis = 'global',
        normalize_input_target = (0, 1),
        normalize_input_clip = True,
        optimizer = 'Adam',
        optimizer_params = {'lr': 0.001},
        n_epochs = n_epochs,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = batch_size)
  )
  algorithms.append(alg)

for prediction_var in ['pi']:
  alg = (
    "vae_torch_" + prediction_var + "_similar",
    "Variational autoencoder",
    lambda input_dim, repr_dim, prediction_var=prediction_var:
      __import__('models.vae_pytorch').vae_pytorch.VAE().init(
        input_dim = input_dim,
        latent_dim = repr_dim,
        #enc_dims = [],
        enc_dims = [300, 300],
        dec_dims = [300, 300],
        enc_activations = 'relu',
        dec_activations = 'relu',
        prediction_mean_activation = 'sigmoid',
        prediction_var = prediction_var,
        prediction_log_var_min = math.log(0.01**2),
        normalize_input_type = 'minmax',
        normalize_input_axis = 'global',
        normalize_input_target = (0, 1),
        normalize_input_clip = True,
        optimizer = 'Adam',
        optimizer_params = {'lr': 0.0005},
        n_epochs = n_epochs,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = batch_size)
  )
algorithms.append(alg)


for prediction_var in ['gs', 'gi']:
  alg = (
    "vae_" + prediction_var + "_e32_d32_minmax01",
    "Variational autoencoder",
    lambda input_dim, repr_dim, prediction_var=prediction_var:
      __import__('models.vae_pytorch').vae_pytorch.VAE().init(
        input_dim = input_dim,
        latent_dim = repr_dim,
        #enc_dims = [],
        enc_dims = [32*repr_dim],
        dec_dims = 'same',
        enc_activations = 'relu',
        dec_activations = 'relu',
        prediction_mean_activation = 'sigmoid',
        prediction_var = prediction_var,
        prediction_log_var_min = math.log(0.01**2),
        normalize_input_type = 'minmax',
        normalize_input_axis = 'global',
        normalize_input_target = (0, 1),
        normalize_input_clip = True,
        optimizer = 'Adam',
        optimizer_params = {'lr': 0.0005},
        n_epochs = n_epochs,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = batch_size)
  )
  algorithms.append(alg)
  alg = (
    "vae_" + prediction_var + "_e32_d32_meanstd01",
    "Variational autoencoder",
    lambda input_dim, repr_dim, prediction_var=prediction_var:
      __import__('models.vae_pytorch').vae_pytorch.VAE().init(
        input_dim = input_dim,
        latent_dim = repr_dim,
        #enc_dims = [],
        enc_dims = [32*repr_dim],
        dec_dims = 'same',
        enc_activations = 'relu',
        dec_activations = 'relu',
        prediction_mean_activation = None,
        prediction_var = prediction_var,
        prediction_log_var_min = math.log(0.01**2),
        normalize_input_type = 'stddev',
        normalize_input_axis = 'global',
        normalize_input_target = (-1, 1),
        normalize_input_clip = False,
        optimizer = 'Adam',
        optimizer_params = {'lr': 0.0005},
        n_epochs = n_epochs,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = batch_size)
  )
  algorithms.append(alg)
  alg = (
    "vae_" + prediction_var + "_e32_d32_5quantile01",
    "Variational autoencoder",
    lambda input_dim, repr_dim, prediction_var=prediction_var:
      __import__('models.vae_pytorch').vae_pytorch.VAE().init(
        input_dim = input_dim,
        latent_dim = repr_dim,
        #enc_dims = [],
        enc_dims = [32*repr_dim],
        dec_dims = 'same',
        enc_activations = 'relu',
        dec_activations = 'relu',
        prediction_mean_activation = 'sigmoid',
        prediction_var = prediction_var,
        prediction_log_var_min = math.log(0.01**2),
        normalize_input_type = 'quantiles',
        normalize_input_quantile = 0.05,
        normalize_input_axis = 'global',
        normalize_input_target = (0, 1),
        normalize_input_clip = True,
        optimizer = 'Adam',
        optimizer_params = {'lr': 0.0005},
        n_epochs = n_epochs,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = batch_size)
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

  #aux_target = pandas.read_hdf("data/TCGA_cancertype.h5", 'cancer_types')
  #logging.info(" * auxiliary target size: %d" % aux_target.shape)

  #common_samples = data.index.intersection(aux_target.index)
  #data = data.loc[common_samples]
  #aux_target = aux_target.loc[common_samples]
  #logging.info(" * number of common samples: %d" % common_samples.size)

  from common import categorical_to_binary

  x = data.as_matrix()
  #y = categorical_to_binary(aux_target.values)
  #num_classes = y.shape[1]

  #x = x[:,0:2000]

  # normalize the input to _total_ unit variance and per-feature zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
    x -= np.mean(x, axis=0)
  
  # FIXME!
  #x = (x - np.amin(x,axis=0)) / (np.amax(x,axis=0) - np.amin(x,axis=0))
  #x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))

  # init rng  
  np.random.seed(seed)
  import torch
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  #if args.cuda ?????:
  #  torch.cuda.manual_seed(seed)
  
  # separate validation set if needed
  val_x = None
  #val_y = None
  if validation_split:
    logging.info("Splitting into training and validation sets")
    m = x.shape[0]
    perm = np.random.permutation(m)
    x = x[perm,:]
    #y = y[perm,:]
    split_point = int(validation_split * m)
    (val_x, x) = (x[:split_point,:], x[split_point:,:])
    #(val_y, y) = (y[:split_point,:], y[split_point:,:])
    logging.info(" * training set shape: %d x %d" % x.shape)
    logging.info(" * validation set shape: %d x %d" % val_x.shape)
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  logging.info("Running the algorithm...")
  logging.info(" * learning a representation of size %d", repr_dim)
  start_time = time.time()
  
  # init the algorithm
  #alg = make_alg(data_dim, repr_dim, num_classes)
  alg = make_alg(data_dim, repr_dim)
  
  # create output dir if does not exist
  ensure_dir_exists('res')

  full_model_id = "%s-%d-%s-s%d%s" % (data_set, repr_dim, alg_id, seed, id_suffix)

  # define the progress saving function
  progress_filename = 'res/progress-encdec-mse-%s.txt' % (full_model_id)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  #aux_progress_filename = 'res/progress-aux-ce-%s.txt' % (full_model_id)
  #aux_progress_file = open(aux_progress_filename, 'w', encoding='utf-8')
  if val_x is not None:
    val_progress_filename = 'res/progress-encdec-validation-mse-%s.txt' % (full_model_id)
    val_progress_file = open(val_progress_filename, 'w', encoding='utf-8')
    #aux_val_progress_filename = 'res/progress-aux-validation-ce-%s.txt' % (full_model_id)
    #aux_val_progress_file = open(aux_val_progress_filename, 'w', encoding='utf-8')
  def save_progress():
    x_pred = alg.decode(alg.encode(x))
    rel_mse = relative_mean_squared_error(x, x_pred)
    progress_file.write("%g\n" % rel_mse)
    #aux_pred = alg.predict_secondary(x)
    #aux_rel_ce = relative_cross_entropy(y, aux_pred)
    #aux_progress_file.write("%g\n" % aux_rel_ce)
    if val_x is not None:
      val_x_pred = alg.decode(alg.encode(val_x))
      rel_mse = relative_mean_squared_error(val_x, val_x_pred)
      val_progress_file.write("%g\n" % rel_mse)
      #val_aux_pred = alg.predict_secondary(val_x)
      #aux_rel_ce = relative_cross_entropy(val_y, val_aux_pred)
      #aux_val_progress_file.write("%g\n" % aux_rel_ce)
  
  # fit to the training data
  alg.learn(x, validation_data=val_x,
            log_file_prefix=("log/%s" % (full_model_id)),
            per_epoch_callback_funs=[save_progress],
            deadline=deadline, max_duration=max_duration)

  # test reconstruction error
  x_pred = alg.decode(alg.encode(x))
  rel_mse = relative_mean_squared_error(x, x_pred)
  val_x_pred = alg.decode(alg.encode(val_x))
  val_rel_mse = relative_mean_squared_error(val_x, val_x_pred)
  logging.info(" * final error: rel_mse = %g, val_rel_mse = %g",
               rel_mse, val_rel_mse)

  elapsed = time.time() - start_time
  logging.info(" * running time = %s", pretty_duration(elapsed))
  
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

