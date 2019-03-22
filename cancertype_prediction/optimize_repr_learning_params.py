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

# list of cancer type pairs that will for the private dataset
# (and will be predicted/classified)
#priv_splits = [
#  ["brain lower grade glioma", "glioblastoma multiforme"],  # brain
#  ["skin cutaneous melanoma", "uveal melanoma"],  # melanoma (skin, eye)
#  ["kidney chromophobe", "kidney clear cell carcinoma", "kidney papillary cell carcinoma"],  # kidney
#  ["lung adenocarcinoma", "lung squamous cell carcinoma"],  # lung
#]

priv_cancertypes = ("lung squamous cell carcinoma", "head & neck squamous cell carcinoma")

val_cancer_type_splits = [
#  ("lung squamous cell carcinoma", "head & neck squamous cell carcinoma"),
  ("kidney clear cell carcinoma", "kidney papillary cell carcinoma"),
#  ("lung adenocarcinoma", "lung squamous cell carcinoma"),
#  ("breast invasive carcinoma", "lung squamous cell carcinoma"),
#  ("colon adenocarcinoma", "rectum adenocarcinoma"),
]
#priv_splits = 'all'

# size of the representation to be learned
#repr_dims = [2, 4, 8, 12, 16]
#repr_dims = [2, 4, 8, 16]
#repr_dims = [16]
#repr_dim = 4

# RNG seeds
#seeds = [0, 1, 2, 3, 4]
#seeds = [0, 1, 2]
#seeds = [4]
seeds = [0]

## parameters for representation learning

#n_epochs = 1
n_epochs = 2000
#deadline = None
#repr_max_duration = None
#repr_max_duration = datetime.timedelta(minutes=1)
repr_max_duration = datetime.timedelta(hours=1)

batch_size = 64

normalize_data = False
#validation_split = 0
validation_split = 0.2

# fraction of data to use for prediction testing (instead of training)
#pred_test_size = 0.3
pred_cv_folds = 10

early_stopping = True
#early_stopping = 'val_secondary_loss'

# logging settings (what to log)
log_weights = False

# droputs (note: must be multiples of 0.1)
#first_layer_dropout = 0.1; other_layer_dropouts = 0.2
#first_layer_dropout = 0.2; other_layer_dropouts = 0.4

# save predicted values (i.e. first encoded then decoded ) in to a file?
save_pred = False

## parameters for predictions

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

## the parameters that will be optimized

domain = [
  {'name': 'learning_rate_log10', 'type': 'continuous', 'domain': (-5,-2)},
  {'name': 'n_hidden_layers', 'type': 'discrete', 'domain': [1, 2, 3]},
#  {'name': 'repr_dim', 'type': 'continuous', 'domain': (1, 10)},
  {'name': 'repr_dim', 'type': 'discrete', 'domain': [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]},
  {'name': 'hidden_layer_size_mul_log10', 'type': 'continuous', 'domain': (0,3)},
  ]
# any constraints for the parameters that will be optimized
constraints = None

## parameters for parameter optimization

gpyopt_batch_size = 4
gpyopt_max_iter = 50
#gpyopt_max_duration = None
#gpyopt_max_duration = datetime.timedelta(minutes=5)
gpyopt_max_duration = datetime.timedelta(hours=12)
gpyopt_deadline = None

# substitute result given to gpyopt in the case of failures
gpyopt_fail_res = 0.5


## other parameters

id_suffix = ""


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

def get_params(params, domain):
  p = dict()
  for i, var in enumerate(domain):
    p[var['name']] = params[i]
  return SimpleNamespace(**p)

def run_optimization(args, domain, constraints, batch_size, max_iter, max_duration=None, deadline=None):
  logging.info('Starting parameter optimization...')
  import GPyOpt
  ensure_dir_exists("param_opt")

  if max_duration is not None:
    new_dl = datetime.datetime.now() + max_duration
    if deadline is None or new_dl < deadline:
      deadline = new_dl

  initial_design_type = 'random'
  initial_design_numdata = batch_size
  logging.info('Selecting initial parameters...')
  space = GPyOpt.core.task.space.Design_space(domain, constraints)
  params = GPyOpt.experiment_design.initial_design(initial_design_type, space, initial_design_numdata)
  logging.info('Running...')
  results = run_batch(args, params)
  all_params = params
  all_results = results
  for i in range(max_iter):
    print(np.hstack((all_params, all_results)), flush=True)
    logging.info('Selecting a new set of parameters...')
    bo = GPyOpt.methods.BayesianOptimization(f=None,
                                              domain = domain,
                                              X = all_params,
                                              Y = -all_results,
                                              acquisition_type = 'EI',
                                              normalize_Y = True,
                                              evaluator_type = 'local_penalization',
                                              batch_size = batch_size,
                                              acquisition_jitter = 0,
                                              maximize = False)

    params = bo.suggest_next_locations()
    logging.info('Running...')
    results = run_batch(args, params)
    all_params = np.vstack((all_params, params))
    all_results = np.vstack((all_results, results))
    np.save("param_opt/opt_params%s.npy" % id_suffix, all_params)
    np.save("param_opt/opt_results%s.npy" % id_suffix, all_results)

    if datetime.datetime.now() >= deadline:
      logging.info('Gpyopt iteration %d: Time based stopping' % (i))
      break
  
  return all_params[np.argmax(np.results)]


def run_batch(args, params):
  ensure_dir_exists("run_parameters")
  params = [get_params(p, domain) for p in params]
  np.save("run_parameters/params.npy", params)
  assert len(params) == gpyopt_batch_size
  args.wait = True
  batch.run_tasks(args)
  # get results
  #return np.random.randn(gpyopt_batch_size, 1)
  res = np.zeros((gpyopt_batch_size, 1))
  for param_id in range(gpyopt_batch_size):
    tot_res = 0
    for val_cancertypes in val_cancer_type_splits:
      data_name = '-'.join(val_cancertypes).replace(' ', '_')
      for seed in seeds:
        #full_model_id = "%s-%d-%s-s%d%s" % (data_name, repr_dim, param_id, seed, id_suffix)
        full_model_id = "%s-%s-%d%s" % (param_id, data_name, seed, id_suffix)
        filename = "param_opt/opt_result-%s.txt" % (full_model_id)
        try:
          tot_res += np.loadtxt(filename)
          import os
          os.remove(filename)
        except:
          tot_res += gpyopt_fail_res
          logging.info('Warning, could not load "%s"' % filename)
    res[param_id] = tot_res / (len(val_cancer_type_splits) * len(seeds))
  return res

def run_test(args, params):
  logging.info('Running final tests with...')
  ensure_dir_exists("run_parameters")
  np.save("run_parameters/test_params.npy", params)
  args.wait = True
  batch.run_tasks(args)


# the task function that is run with each argument combination
def task(args):
  param_id, val_cancertypes, seed = args
  logging.info("val classes = %s, params_id = %s, seed = %d",
               val_cancertypes, param_id, seed)
  #repr_dim, (alg_id, _, make_alg), seed = args
  #logging.info("algorithm = %s, seed = %d", alg_id, seed)
  # read the data sets
  alg_id = param_id
  logging.info("Loading parameters...")
  params = np.load("run_parameters/params.npy")
  params = params[param_id]
  logging.info(" * %s" % params)

  (gene_expr, cancer_type) = load_data()

  # split
  logging.info("Splitting...")
  true_priv = cancer_type.isin(priv_cancertypes)
  priv = cancer_type.isin(val_cancertypes)
  pub = np.logical_and(~priv, ~true_priv)

  logging.info(" * %d true private samples, %d validation private samples, %d public samples (of %d total)" %
              (sum(true_priv), sum(priv), sum(pub), priv.size))

  from common import categorical_to_binary

  x_pub = gene_expr[pub].as_matrix()
  y_pub = cancer_type[pub].cat.codes.as_matrix()
  x_priv = gene_expr[priv].as_matrix()
  y_priv = cancer_type[priv].cat.codes.as_matrix()
  #y = categorical_to_binary(aux_target.values)
  #num_classes = y.shape[1]

  data_name = '-'.join(val_cancertypes).replace(' ', '_')

  # A hack to have a different seed if the algorithm is run multiple times
  # with the same parameters. Destroys reproducibility...
  import time
  seed0 = int(time.time()*100) % (2**32)
  # init rng  
  np.random.seed(seed0)
  import torch
  torch.manual_seed(seed0)
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    torch.cuda.manual_seed(seed0)

  #full_model_id = "%s-%d-%s-s%d%s" % (data_name, repr_dim, alg_id, seed, id_suffix)
  full_model_id = "%s-%s-%d%s" % (param_id, data_name, seed, id_suffix)

  acc = run_alg(x_pub, y_pub, x_priv, y_priv, params, full_model_id)

  logging.info("Writing results to disk...")
  filename = "param_opt/opt_result-%s.txt" % (full_model_id)
  logging.info(" * filename: %s", filename)
  with open(filename, 'w', encoding='utf-8') as f:
    f.write("%.6f\n" % acc)


def load_data():
  import pandas
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

  return (gene_expr, cancer_type)


def run_alg(x_pub, y_pub, x_priv, y_priv, params, full_model_id):

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
    logging.info(" * training set shape: %d x %d" % x.shape)
    logging.info(" * validation set shape: %d x %d" % val_x.shape)
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  repr_dim = int(round(params.repr_dim))

  logging.info("Learning the representation on public data...")
  logging.info(" * learning a representation of size %d", repr_dim)
  start_time = time.time()
  
  # init the algorithm
  #alg = make_alg(data_dim, repr_dim, num_classes)
  #alg = make_alg(data_dim, repr_dim)
  from models.vae_pytorch import VAE
  alg = VAE().init(
    input_dim = data_dim,
    latent_dim = repr_dim,
    #enc_dims = [],
    enc_dims = [int(10 ** params.hidden_layer_size_mul_log10)*repr_dim] * int(params.n_hidden_layers),
    dec_dims = 'same',
    enc_activations = 'relu',
    dec_activations = 'relu',
    prediction_mean_activation = 'sigmoid',
    prediction_var = 'gs',
    prediction_log_var_min = math.log(0.01**2),
    normalize_input_type = 'quantiles',
    normalize_input_quantile = 0.05,
    normalize_input_axis = 'global',
    normalize_input_target = (0, 1),
    normalize_input_clip = True,
    optimizer = 'Adam',
    optimizer_params = {'lr': 10.0 ** params.learning_rate_log10},
    n_epochs = n_epochs,
    early_stopping = True,
    reduce_lr_on_plateau = False,
    batch_size = batch_size)

  # create output dir if does not exist
  #ensure_dir_exists('res')

  # define the progress saving function
  ensure_dir_exists('param_opt/progress')
  progress_filename = 'param_opt/progress/encdec-mse-%s.txt' % (full_model_id)
  progress_file = open(progress_filename, 'w', encoding='utf-8')
  #aux_progress_filename = 'param_opt/progress/aux-ce-%s.txt' % (full_model_id)
  #aux_progress_file = open(aux_progress_filename, 'w', encoding='utf-8')
  if val_x is not None:
    val_progress_filename = 'param_opt/progress/encdec-validation-mse-%s.txt' % (full_model_id)
    val_progress_file = open(val_progress_filename, 'w', encoding='utf-8')
    #aux_val_progress_filename = 'param_opt/progress/aux-validation-ce-%s.txt' % (full_model_id)
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
  ensure_dir_exists("param_opt/log/")
  alg.learn(x, validation_data=val_x,
            log_file_prefix=("param_opt/log/%s" % (full_model_id)),
            per_epoch_callback_funs=[save_progress],
            deadline=None, max_duration=repr_max_duration)

  # test reconstruction error
  x_pred = alg.decode(alg.encode(x))
  rel_mse = relative_mean_squared_error(x, x_pred)
  val_x_pred = alg.decode(alg.encode(val_x))
  val_rel_mse = relative_mean_squared_error(val_x, val_x_pred)
  logging.info(" * final error: rel_mse = %g, val_rel_mse = %g",
              rel_mse, val_rel_mse)

  elapsed = time.time() - start_time
  logging.info(" * running time = %s", pretty_duration(elapsed))


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

  ##################################
  #   prediction
  #################################

  x = x_repr

  # private or non-private logistic regression
  private = True

  # test prediction with cross validation
  logging.info("Prediction with %d-fold cross validation...", pred_cv_folds)
  from sklearn.model_selection import StratifiedKFold
  cv = StratifiedKFold(n_splits=pred_cv_folds, shuffle=True, random_state=0)
  avg_test_acc = 0
  for fold, (train, test) in enumerate(cv.split(x, y)):
    logging.info("Fold %d...", fold)
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
  
    # init rng  
    #np.random.seed(seed0)

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
    avg_test_acc += test_acc
  
  avg_test_acc /= pred_cv_folds
  logging.info("Average test accuracy = %.6f", avg_test_acc)
  
  return avg_test_acc
  
########## MAIN ##########

def main():
  
  param_ids = range(gpyopt_batch_size)
  batch.init(task=task, args_ranges=(param_ids, val_cancer_type_splits, seeds))

  args = batch.parse_args()
  if args.action == "task":
    assert args.task is not None
    id = int(args.task)
    logging.info('Running task id %d...', id)
    batch._run_task(id)
  else:
    best_params = run_optimization(args, domain, constraints, gpyopt_batch_size, gpyopt_max_iter, max_duration=gpyopt_max_duration, deadline=gpyopt_deadline)


# init and run
main()

