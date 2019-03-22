'''
Representation learning testing for TCGA data.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
import datetime

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
target_set = "TCGA_split_priv_cancertype"

# fraction of data to use for testing (instead of training)
test_size = 0.3

# size of the representation
repr_dims = [2, 4, 8, 12, 16]
#repr_dims = [16]

algorithms = []

seeds = [0, 1, 2, 3, 4]
#seeds = [0, 1, 2]
#seeds = [3, 4]

#id_suffix = "-ess" # early stopping secondary (loss)
id_suffix = ""

dim_red_algs = [
  'rand_proj',
  'pca',
  'vae_torch_gs_e32_d32',
  'vae_torch_gi_e32_d32',
  'vae_torch_ps_e32_d32',
  'vae_torch_pi_e32_d32',
  'vae_torch_pi_similar',
  'vae_gs_e32_d32_minmax01',
  'vae_gi_e32_d32_minmax01',
  'vae_gs_e32_d32_meanstd01',
  'vae_gi_e32_d32_meanstd01',
  'vae_gs_e32_d32_5quantile01',
  'vae_gi_e32_d32_5quantile01',
]

#scale_fun = "none"; scale_const = 1.0
#scale_fun = "norm_max"; scale_const = 1.01
#scale_fun = "dims_max"; scale_const = 1.0
scale_fun = "norm_avg"; scale_const = 1.0
#scale_fun = "dims_std"; scale_const = 1.0
#clip = "none"
clip = "norm"
#clip = "dims"
bounding_slack = 0.01
private = True
#epsilon = np.inf
epsilon = 1.0
regularizer_strength = 0.1

#if private:
#  id_suffix = "-e%g" % (epsilon)
#else:
#  id_suffix = "-nonpriv"

##########################################################################################
# END OF SETUP
##########################################################################################


# the task function that is run with each argument combination
def task(args):
  repr_dim, alg_id, seed = args
  logging.info("representation size = %d, algorithm = %s, seed = %d", repr_dim, alg_id, seed)

  # read the PADS gene expression data
  logging.info("Reading reduced gene expression data...")
  filename = ("data_repr/repr-%s-%d-%s-%s-s%d%s.csv" %
             (data_set, repr_dim, aux_data_set, alg_id, seed, id_suffix))
  logging.info(" * filename: %s" % filename)
  x = np.loadtxt(filename,
             delimiter=',')
  if x.ndim < 2:
    x = x[:,np.newaxis]
  logging.info(" * data shape: %d x %d" % x.shape)

  logging.info("Reading cancer types...")
  filename = "data/%s.h5" % (target_set)
  logging.info(" * filename: %s" % filename)
  import pandas
  target = pandas.read_hdf(filename, 'cancer_types')
  logging.info(" * target size: %d" % target.shape)
  #y = target.as_matrix()
  y = target.cat.codes.as_matrix()
  
  # split train and test sets
  logging.info("Splitting to train and test sets...")
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
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
              (repr_dim, aux_data_set, alg_id, seed, scale_fun, scale_const, clip,
              ("-e%g" % (epsilon) if private else "-nonpriv")))
  logging.info(" * filename: %s", filename)
  with open(filename, 'w', encoding='utf-8') as f:
    f.write("%.6f\n" % test_acc)
  

########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(repr_dims, dim_red_algs, seeds))
batch.main()
