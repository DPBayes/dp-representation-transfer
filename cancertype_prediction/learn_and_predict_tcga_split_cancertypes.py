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
#data_set = "TCGA_split_pub_geneexpr"
data_set = "TCGA_geneexpr"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'
target_set = "TCGA_cancertype"
target_type = 'cancer_types'

#priv_splits = [
#  ["brain lower grade glioma", "glioblastoma multiforme"],  # brain
#  ["skin cutaneous melanoma", "uveal melanoma"],  # melanoma (skin, eye)
#  ["kidney chromophobe", "kidney clear cell carcinoma", "kidney papillary cell carcinoma"],  # kidney
#  ["lung adenocarcinoma", "lung squamous cell carcinoma"],  # lung
#]
priv_splits = [
  ("lung squamous cell carcinoma", "head & neck squamous cell carcinoma"),
  ("kidney clear cell carcinoma", "kidney papillary cell carcinoma"),
  ("lung adenocarcinoma", "lung squamous cell carcinoma"),
  ("breast invasive carcinoma", "lung squamous cell carcinoma"),
  ("colon adenocarcinoma", "rectum adenocarcinoma"),
]


priv_splits = 'all'

# size of the representation to be learned
#repr_dims = [2, 4, 8, 12, 16]
#repr_dims = [2, 4, 8, 16]
repr_dims = [8]
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

# fraction of data to use for prediction testing (instead of training)
pred_test_size = 0.3

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
#seeds = [0, 1, 2, 3, 4]
seeds = [0, 1, 2]
#seeds = [4]
#seeds = [0]

#id_suffix = "-ess" # early stopping secondary (loss)
id_suffix = ""


#scale_fun = "none"; scale_const = 1.0
#scale_fun = "norm_max"; scale_const = 1.01
#scale_fun = "dims_max"; scale_const = 1.0
scale_fun = "norm_avg"; scale_const = 1.0
#scale_fun = "dims_std"; scale_const = 1.0
#clip = "none"
clip = "norm"
#clip = "dims"
bounding_slack = 0.01
#private = False
#epsilon = np.inf
epsilon = 1.0
regularizer_strength = 0.1


algorithms = []

# Random projection
'''alg = (
  "rand_proj",
  "Random projection (Gaussian)",
  lambda input_dim, repr_dim:
    __import__('models.rand_proj').rand_proj.RandomProjection().init(
      input_dim = input_dim,
      output_dim = repr_dim)
)
algorithms.append(alg)'''

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

'''for prediction_var in ['gs', 'gi', 'ps', 'pi']:
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
  #algorithms.append(alg)

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
#algorithms.append(alg)


#for prediction_var in ['gs', 'gi']:
for prediction_var in ['gs']:
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
  algorithms.append(alg)'''



if algs_to_run != "all":
  algorithms = [a for a in algorithms if (a[0] in algs_to_run)]

if priv_splits == "all":
  cancer_type_values = ['breast invasive carcinoma', 'kidney clear cell carcinoma',
       'lung adenocarcinoma', 'ovarian serous cystadenocarcinoma',
       'lung squamous cell carcinoma', 'glioblastoma multiforme',
       'head & neck squamous cell carcinoma',
       'uterine corpus endometrioid carcinoma', 'stomach adenocarcinoma',
       'thyroid carcinoma', 'prostate adenocarcinoma', 'colon adenocarcinoma',
       'brain lower grade glioma', 'skin cutaneous melanoma',
       'liver hepatocellular carcinoma', 'bladder urothelial carcinoma',
       'kidney papillary cell carcinoma', 'cervical & endocervical cancer',
       'sarcoma', 'esophageal carcinoma', 'acute myeloid leukemia',
       'pancreatic adenocarcinoma', 'pheochromocytoma & paraganglioma',
       'rectum adenocarcinoma', 'testicular germ cell tumor', 'thymoma',
       'adrenocortical cancer', 'kidney chromophobe', 'mesothelioma',
       'uveal melanoma', 'uterine carcinosarcoma',
       'diffuse large B-cell lymphoma', 'cholangiocarcinoma']
  from itertools import combinations
  priv_splits = list(combinations(cancer_type_values, 2))

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
  return np.average((x - x_pred) ** 2)

def relative_mean_squared_error(x, x_pred):
  mse = mean_squared_error(x, x_pred)
  x_avg = np.average(x, axis=0)
  return mse / np.average((x - x_avg) ** 2)

# the task function that is run with each argument combination
def task(args):
  import pandas
  priv_cancertypes, repr_dim, (alg_id, _, make_alg), seed = args
  logging.info("priv classes = %s, algorithm = %s, seed = %d",
               priv_cancertypes, alg_id, seed)
  #repr_dim, (alg_id, _, make_alg), seed = args
  #logging.info("algorithm = %s, seed = %d", alg_id, seed)
  # read the data sets
  logging.info("Reading data...")
  gene_expr = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  logging.info(" * gene expression shape: %d x %d" % gene_expr.shape)

  logging.info("Filtering out genes with low expressions...")
  low_expr = (np.median(gene_expr, axis=0) < 0.0)
  gene_expr = gene_expr.iloc[:,~low_expr]
  logging.info(" * %d of %d remaining (%d removed)" %
              (sum(~low_expr), low_expr.size, sum(low_expr)))

  logging.info("Loading cancer types...")
  cancer_type = pandas.read_hdf("data/%s.h5" %  (target_set), target_type)
  assert np.array_equal(gene_expr.index, cancer_type.index)

  #for priv_cancertypes in priv_splits:
  for unused in [1]:
    # split
    logging.info("Splitting...")
    priv = cancer_type.isin(priv_cancertypes)
    logging.info(" * %d private samples, %d public samples (of %d total)" %
                (sum(priv), sum(~priv), priv.size))

    from common import categorical_to_binary

    x_pub = gene_expr[~priv].as_matrix()
    y_pub = cancer_type[~priv].cat.codes.as_matrix()
    x_priv = gene_expr[priv].as_matrix()
    y_priv = cancer_type[priv].cat.codes.as_matrix()
    #y = categorical_to_binary(aux_target.values)
    #num_classes = y.shape[1]

    data_name = '-'.join(priv_cancertypes).replace(' ', '_')

    # init rng  
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      torch.cuda.manual_seed(seed)

    ##################################
    #   representation learning
    #################################
    x = x_pub
    y = y_pub

    # separate validation set if needed
    val_x = None
    #val_y = None
    if validation_split:
      logging.info("Splitting into training and validation sets")
      from sklearn.model_selection import train_test_split
      train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=validation_split, random_state=0)
      x, y = train_x, train_y
      #m = x.shape[0]
      #perm = np.random.permutation(m)
      #x = x[perm,:]
      #y = y[perm,:]
      #split_point = int(validation_split * m)
      #(val_x, x) = (x[:split_point,:], x[split_point:,:])
      #(val_y, y) = (y[:split_point,:], y[split_point:,:])
      logging.info(" * training set shape: %d x %d" % x.shape)
      logging.info(" * validation set shape: %d x %d" % val_x.shape)
    
    data_dim = x.shape[1]
    logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

    logging.info("Learning the representaiton on public data...")
    logging.info(" * learning a representation of size %d", repr_dim)
    start_time = time.time()
    
    # init the algorithm
    #alg = make_alg(data_dim, repr_dim, num_classes)
    alg = make_alg(data_dim, repr_dim)
    
    # create output dir if does not exist
    ensure_dir_exists('res')

    full_model_id = "%s-%d-%s-s%d%s" % (data_name, repr_dim, alg_id, seed, id_suffix)

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
    #logging.info("Saving the learned model...")
    #ensure_dir_exists('repr_models')
    #alg.save("repr_models/%s" % (full_model_id))

    ##################################
    #   representation mapping
    #################################

    x = x_priv
    y = y_priv

    # get the representation
    logging.info("Making the representation of private data...")
    x_repr = alg.encode(x)

    # test to predict the data itself
    x_pred = alg.decode(x_repr)
    rel_mse = relative_mean_squared_error(x, x_pred)
    logging.info(" * reconstruct the data: rel_mse = %g", rel_mse)
    ensure_dir_exists("res")
    with open("res/private-encdec-rel_mse-%d-%s-%s-s%d%s.txt" %
              (repr_dim, data_name, alg_id, seed, id_suffix),
              'w', encoding='utf-8') as f:
      f.write("%.6f\n" % rel_mse)

    # save the representation
    #logging.info("Saving the representation...")
    #ensure_dir_exists("data_repr")
    #np.savetxt("data_repr/repr-%s-%d-%s-s%d%s.csv" %
    #           (data_name, repr_dim, alg_id, seed, id_suffix),
    #           x_repr, delimiter=',')


    ##################################
    #   prediction
    #################################

    x = x_repr

    # split train and test sets
    logging.info("Splitting to train and test sets...")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=pred_test_size, random_state=0)
    logging.info(" * train samples: %d" % x_train.shape[0])
    logging.info(" * test samples: %d" % x_test.shape[0])
      
    # init rng  
    np.random.seed(seed)

    #print(np.amax(np.linalg.norm(x_train, axis=1)))
    #print(np.mean(np.linalg.norm(x_train, axis=1)))

    logging.info("Bounding the data to 1-sphere...")
    if scale_fun == "norm_max":
      logging.info(" * scale by max norm")
      scale_factor = np.amax(np.linalg.norm(x_train, axis=1))
    elif scale_fun == "dims_max":
      logging.info(" * scale each dimension by max absolute value")
      scale_factor = np.amax(np.abs(x_train), axis=0)
    elif scale_fun == "norm_avg":
      logging.info(" * scale by average norm")
      scale_factor = np.mean(np.linalg.norm(x_train, axis=1))
    elif scale_fun == "dims_std":
      logging.info(" * scale each dimension by standard deviation")
      scale_factor = np.std(x_train, axis=0)
    elif scale_fun == "none":
      scale_factor = 1.0
    else:
      assert False

    x_train /= scale_factor * scale_const
    x_test /= scale_factor * scale_const
    #print(np.amax(np.linalg.norm(x_train, axis=1, keepdims=True)))
    if clip == "norm":
      logging.info(" * clip norms to max 1")
      x_train /= np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True) * (1 + bounding_slack), 1)
      x_test /= np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True) * (1 + bounding_slack),1)
    elif clip == "dims":
      assert False, "not implemented"
    elif clip == "none":
      logging.info(" * no clipping -> no bounding")
      assert private == False #or np.isinf(epsilon)
    else:
      assert False

    for private in [False, True]:
      # fit
      logging.info("Fitting a model...")
      if private:
        logging.info(" * DP logistic regression: epsilon=%g, alpha=%g", epsilon, regularizer_strength)
        from models.logistic_regression import DPLogisticRegression
        model = DPLogisticRegression().init(repr_dim, classes=np.unique(y),
                                            alpha=regularizer_strength, epsilon=epsilon)
      else:
        logging.info(" * logistic regression: alpha=%g", regularizer_strength)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1/regularizer_strength)
      
      model.fit(x_train, y_train)
      #print(model.predict(x_test))

      # compute mean accuracy on test set
      logging.info("Testing the model...")
      #acc = model.score(x_test, y_test)
      from sklearn.metrics import accuracy_score
      train_acc = accuracy_score(y_train, model.predict(x_train))
      test_acc = accuracy_score(y_test, model.predict(x_test))
      logging.info(" * train accuracy = %.6f", train_acc)
      logging.info(" * test accuracy = %.6f", test_acc)
      
      logging.info("Writing results to disk...")
      ensure_dir_exists("res")
      filename = ("res/cancertype-pred-accuracy-%d-%s-%s-s%d-%s-%d-%s%s.txt" %
                  (repr_dim, data_name, alg_id, seed, scale_fun, scale_const, clip,
                  ("-e%g" % (epsilon) if private else "-nonpriv")))
      logging.info(" * filename: %s", filename)
      with open(filename, 'w', encoding='utf-8') as f:
        f.write("%.6f\n" % test_acc)

########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(priv_splits, repr_dims, algorithms, seeds))
#batch.init(task=task, args_ranges=(repr_dims, algorithms, seeds))
batch.main()


# try to workaround a bug that tensorflow randomly throws an exception in the end
# this seems to be the same: https://github.com/tensorflow/tensorflow/issues/3745
# possibly also this: https://github.com/tensorflow/tensorflow/issues/3388
from sys import modules
if "keras.backend.tensorflow_backend" in modules:
  import keras.backend
  keras.backend.clear_session()

