'''
Representation learning testing for TCGA data.
Set all parameters in this script & run with parameter "batch" or "local".
'''

import numpy as np
import logging
import datetime
import time
import math
import pickle


from common import expInterp, ensure_dir_exists, pretty_duration
import batch2
from types import SimpleNamespace

######################################################################
# SETUP
######################################################################
# set the following parameter values

# logging configuration
logging.basicConfig(level=logging.INFO)
log_file_formatter = logging.Formatter(
                       fmt="%(asctime)s - %(levelname)s (%(name)s): %(message)s",
                       datefmt="%Y-%m-%d %H:%M:%S")

# input dataset
#data_set = "TCGA_split_pub_geneexpr"
data_set = "TCGA_geneexpr"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'
target_set = "TCGA_cancertype"
target_type = 'cancer_types'

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
cancer_type_pairs = list(combinations(cancer_type_values, 2))


## general test parameters

test_params = SimpleNamespace(
  # selection of private and validation_private cancertypes
  #priv_cancertypes = cancer_type_pairs[0],
  #priv_cancertypes = cancer_type_pairs[1],
  #priv_cancertypes = cancer_type_pairs[2],
  #priv_cancertypes = cancer_type_pairs[3],
  
  # RNG seeds
  main_seed = 0,
  test_seeds = [0, 1, 2, 3, 4],

  # representation learning algorithm
  #repr_alg = "VAE",
  #repr_alg = "rand_proj",
  repr_alg = "PCA",

  repr_dims = list(range(2, 20))
)

# slurm args

slurm_cpu_args = {
    "time": "2:00:00",
    "mem": "16G",
    "partition": "short",
    "constraint": "hsw",
    "cpus-per-task": "1",
}

slurm_gpu_args = {
    "time": "2:00:00",
    "mem": "16G",
    "partition": "gpushort",
    "gres": "gpu:1",
    "constraint": "hsw",
    "cpus-per-task": "1",
}


## parameters for parameter optimization

## other parameters

test_id = "rdbf-"

## model definitions and model-specific parameters for representation learning
def select_repr_alg(repr_alg):
  if repr_alg == "PCA":
    # fixed parameters
    fixed_params = SimpleNamespace(
      repr_learn_max_duration = None,
      repr_learn_validation_split = 0,
    )
    # the parameters that will be optimized
    opt_param_domain = []
    # any constraints for the parameters that will be optimized
    opt_param_constraints = []
    # alg construction
    def make_alg(data_dim, repr_dim, params):
      from models.pca import PCA
      alg = PCA().init(
        input_dim = data_dim,
        output_dim = repr_dim)
      return alg
    slurm_args = slurm_cpu_args
  
  elif repr_alg == "rand_proj":
    # fixed parameters
    fixed_params = SimpleNamespace(
      repr_learn_max_duration = None,
      repr_learn_validation_split = 0,
    )
    # the parameters that will be optimized
    opt_param_domain = []
    # any constraints for the parameters that will be optimized
    opt_param_constraints = []
    # alg construction
    def make_alg(data_dim, repr_dim, params):
      from models.rand_proj import RandomProjection
      alg = RandomProjection().init(
        input_dim = data_dim,
        output_dim = repr_dim)
      return alg
    slurm_args = slurm_cpu_args
  
  else:
    assert False, "invalid alg"
  
  return (fixed_params, opt_param_domain, opt_param_constraints, make_alg, slurm_args)


## parameters for predictions

pred_params = SimpleNamespace(
  # fraction of data to use for prediction testing (instead of training)
  #pred_test_size = 0.3,
  pred_cv_folds = 10,

  # how to scale data before clipping to 1-ball
  #pred_scale_fun = "none", pred_scale_const = 1.0,
  #pred_scale_fun = "norm_max", pred_scale_const = 1.01,
  #pred_scale_fun = "dims_max", pred_scale_const = 1.0,
  pred_scale_fun = "norm_avg", pred_scale_const = 1.0,
  #pred_scale_fun = "dims_std", pred_scale_const = 1.0,

  # how to clip the (scaled) data to 1-ball
  #pred_clip = "none",
  pred_clip = "norm",
  #pred_clip = "dims",
  pred_bounding_slack = 0.01,

  #pred_private = False,
  #pred_epsilon = np.inf,

  # DP epsilon
  pred_epsilon = 1.0,

  # weight regularizer
  pred_regularizer_strength = 0.1,
)



################################################################################
# END OF SETUP
################################################################################

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

'''def run_optimization(args, fixed_params, domain, constraints, slurm_args=None):
  logging.info('Starting parameter optimization...')
  ensure_dir_exists("param_opt")

  assert len(domain) == 1
  opt_params = np.array(domain[0]['domain'])[:, np.newaxis]

  opt_seeds = fixed_params.opt_seeds
  n_seeds = len(opt_seeds)
  n_params = len(opt_params)

  opt_seeds = opt_seeds * n_params
  #opt_params = np.tile(opt_params, (n_seeds, 1))
  opt_params = np.repeat(opt_params, n_seeds, axis=0)
  print(opt_seeds)
  print(opt_params)

  logging.info('Running...')
  task_params = [get_params(p, domain) for p in opt_params]
  results = run_optimization_batch(args, fixed_params, task_params, opt_seeds, slurm_args)
  all_params = opt_params
  all_results = results
  np.save("param_opt/opt_params-%s.npy" % (fixed_params.test_name), all_params)
  np.save("param_opt/opt_results-%s.npy" % (fixed_params.test_name), all_results)

  all_params = [get_params(p, domain) for p in all_params]
  all_results = list(all_results)

  filename = "param_opt/paramopt-%s.txt" % (fixed_params.test_name)
  logging.info("Writing params and result to '%s'" % filename)
  with open(filename, 'wb') as f:
    pickle.dump(all_params, f)
    pickle.dump(all_results, f)

  best_params_id = np.argmax(all_results)
  best_params = all_params[best_params_id]
  best_result = all_results[best_params_id]
  logging.info('Final best result: %g', best_result)
  logging.info(' * obtained with: %s', best_params)

  filename = "res/paramopt_best_result-%s.txt" % (fixed_params.test_name)
  logging.info("Writing best result to '%s'" % filename)
  np.savetxt(filename, best_result)

  filename = "param_opt/paramopt_best_params-%s.txt" % (fixed_params.test_name)
  logging.info("Writing best params to '%s'" % filename)
  with open(filename, 'wb') as f:
    pickle.dump(best_params, f)

  return best_params


def run_optimization_batch(args, fixed_params, task_params, seeds, slurm_args=None):
  ensure_dir_exists("run_parameters")
  param_ids = range(len(task_params))
  for param, seed, param_id in zip(task_params, seeds, param_ids):
    param.param_id = param_id
    param.seed = seed
  #np.save("run_parameters/params-%s.npy" % (test_name), params)
  common_params = SimpleNamespace(
    **fixed_params.__dict__,
    pred_cancertypes=fixed_params.priv_cancertypes,
    skip_cancertypes=[],
    task_type='paramopt',
  )
  args.wait = True
  batch2.run_tasks(args, common_params, task_params, slurm_args=slurm_args,
    params_file=("run_parameters/batch-%s.pkl" % (fixed_params.test_name)))
  # get results
  res = np.zeros((len(task_params), 1))
  for param_id in param_ids:
    full_model_id = "%s-%s" % (fixed_params.test_name, param_id)
    filename = "param_opt/opt_result-%s.txt" % (full_model_id)
    try:
      res[param_id] = np.loadtxt(filename)
      import os
      os.remove(filename)
    except:
      res[param_id] = gpyopt_fail_res + np.random.randn() * gpyopt_fail_res_std
      logging.info('Warning, could not load "%s"' % filename)
  return res
'''

def run_test(args, fixed_params, task_param, seeds, slurm_args=None):
  logging.info('Running tests...')

  task_params = [[SimpleNamespace(
      pred_cancertypes=priv_cancertypes,
      seed=seed,
    )
    for seed in seeds]
    for priv_cancertypes in cancer_type_pairs]
  task_params = sum(task_params, [])
  param_ids = range(len(task_params))  

  for param, param_id in zip(task_params, param_ids):
    param.param_id = param_id
  
  ensure_dir_exists("run_parameters")
  #np.save("run_parameters/params-%s.npy" % (test_name), params)
  common_params = SimpleNamespace(
    **fixed_params.__dict__,
  )
  args.wait = True
  batch2.run_tasks(args, common_params, task_params, slurm_args=slurm_args,
    params_file=("run_parameters/batch-%s.pkl" % (fixed_params.test_name)))
  #
  res = np.zeros((len(task_params), 1))
  for param_id in param_ids:
    full_model_id = "%s-%s" % (fixed_params.test_name, param_id)
    filename = "param_opt/opt_result-%s.txt" % (full_model_id)
    try:
      res[param_id] = np.loadtxt(filename)
      import os
      os.remove(filename)
    except:
      res[param_id] = np.nan
      logging.info('Warning, could not load "%s"' % filename)
  res = np.reshape(res, (len(cancer_type_pairs), len(seeds)))
  filename = "res/test_results-%s.txt" % (fixed_params.test_name)
  logging.info("Writing final results to '%s'" % filename)
  np.savetxt(filename, res)


# the task function that is run with each argument combination
def task(common_params, task_params):
  # add logging file
  log_file_name = "log/opttest-task-%s-%s-s%d.log" % (common_params.test_name,
                    common_params.task_type, task_params.seed)
  log_file_handler = logging.FileHandler(log_file_name, mode='w')
  log_file_handler.setFormatter(log_file_formatter)
  logging.getLogger().addHandler(log_file_handler)

  logging.info("test_name = %s", common_params.test_name)
  logging.info("params_id = %s", task_params.param_id)
  logging.info("Running with params: %s" % task_params)
  params = SimpleNamespace(**common_params.__dict__, **task_params.__dict__)

  (gene_expr, cancer_type) = load_data()

  # split
  logging.info("Splitting...")
  priv = cancer_type.isin(params.pred_cancertypes)
  skip = cancer_type.isin(params.skip_cancertypes)
  pub = np.logical_and(~priv, ~skip)

  logging.info(" * %d private samples, %d skipped samples, %d public samples (of %d total)" %
              (sum(priv), sum(skip), sum(pub), priv.size))

  from common import categorical_to_binary

  x_pub = gene_expr[pub].as_matrix()
  y_pub = cancer_type[pub].cat.codes.as_matrix()
  x_priv = gene_expr[priv].as_matrix()
  y_priv = cancer_type[priv].cat.codes.as_matrix()

  seed = int(params.seed)
  # init rng  
  np.random.seed(seed)
  import torch
  torch.manual_seed(seed)
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    torch.cuda.manual_seed(seed)

  full_model_id = "%s-%s" % (common_params.test_name, task_params.param_id)

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
  if params.repr_learn_validation_split:
    logging.info("Splitting into training and validation sets")
    from sklearn.model_selection import train_test_split
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=params.repr_learn_validation_split, random_state=0)
    x, y = train_x, train_y
    logging.info(" * training set shape: %d x %d" % x.shape)
    logging.info(" * validation set shape: %d x %d" % val_x.shape)
  
  data_dim = x.shape[1]
  logging.info(" * data shape after preprocessing: %d x %d" % x.shape)

  repr_dim = int(round(params.repr_dim))

  logging.info("Learning the representation on public data...")
  logging.info(" * learning a representation of size %d", repr_dim)
  start_time = time.time()
  
  (_, _, _, make_alg, _) = select_repr_alg(params.repr_alg)

  # init the algorithm
  #alg = make_alg(data_dim, repr_dim, num_classes)
  #alg = make_alg(data_dim, repr_dim)
  alg = make_alg(data_dim, repr_dim, params)
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
            deadline=None, max_duration=params.repr_learn_max_duration)

  # test reconstruction error
  x_pred = alg.decode(alg.encode(x))
  rel_mse = relative_mean_squared_error(x, x_pred)
  if val_x is not None:
    val_x_pred = alg.decode(alg.encode(val_x))
    val_rel_mse = relative_mean_squared_error(val_x, val_x_pred)
  else:
    val_rel_mse = np.nan
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
  logging.info("Prediction with %d-fold cross validation...", params.pred_cv_folds)
  from sklearn.model_selection import StratifiedKFold
  cv = StratifiedKFold(n_splits=params.pred_cv_folds, shuffle=True, random_state=0)
  avg_test_acc = 0
  for fold, (train, test) in enumerate(cv.split(x, y)):
    logging.info("Fold %d...", fold)
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
  
    # init rng  
    #np.random.seed(seed)

    logging.info("Bounding the data to 1-sphere...")
    if params.pred_scale_fun == "norm_max":
      logging.info(" * scale by max norm")
      scale_factor = np.amax(np.linalg.norm(x_train, axis=1))
    elif params.pred_scale_fun == "dims_max":
      logging.info(" * scale each dimension by max absolute value")
      scale_factor = np.amax(np.abs(x_train), axis=0)
    elif params.pred_scale_fun == "norm_avg":
      logging.info(" * scale by average norm")
      scale_factor = np.mean(np.linalg.norm(x_train, axis=1))
    elif params.pred_scale_fun == "dims_std":
      logging.info(" * scale each dimension by standard deviation")
      scale_factor = np.std(x_train, axis=0)
    elif params.pred_scale_fun == "none":
      scale_factor = 1.0
    else:
      assert False

    x_train /= scale_factor * params.pred_scale_const
    x_test /= scale_factor * params.pred_scale_const
    if params.pred_clip == "norm":
      logging.info(" * clip norms to max 1")
      x_train /= np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True) * (1 + params.pred_bounding_slack), 1)
      x_test /= np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True) * (1 + params.pred_bounding_slack),1)
    elif params.pred_clip == "dims":
      assert False, "not implemented"
    elif params.pred_clip == "none":
      logging.info(" * no clipping -> no bounding")
      assert private == False #or np.isinf(epsilon)
    else:
      assert False

    # fit
    logging.info("Fitting a model...")
    if private:
      logging.info(" * DP logistic regression: epsilon=%g, alpha=%g", params.pred_epsilon, params.pred_regularizer_strength)
      from models.logistic_regression import DPLogisticRegression
      model = DPLogisticRegression().init(repr_dim, classes=np.unique(y),
                                          alpha=params.pred_regularizer_strength, epsilon=params.pred_epsilon)
    else:
      logging.info(" * logistic regression: alpha=%g", params.pred_regularizer_strength)
      from sklearn.linear_model import LogisticRegression
      model = LogisticRegression(C=1/params.pred_regularizer_strength)
    
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
  
  avg_test_acc /= params.pred_cv_folds
  logging.info("Average test accuracy = %.6f", avg_test_acc)
  
  return avg_test_acc


########## MAIN ##########

def main(args):
    data_name = (('-'.join(['priv',] + test_params.priv_cancertypes))
                  .replace(' ', '_').replace('&', '_'))
    test_params.test_name = "%s%s-%s" % (test_id, data_name, test_params.repr_alg)

    # add logging file
    log_file_name = "log/opttest-main-%s.log" % (test_params.test_name)
    log_file_handler = logging.FileHandler(log_file_name, mode='w')
    log_file_handler.setFormatter(log_file_formatter)
    logging.getLogger().addHandler(log_file_handler)

    # init seeds
    import random
    random.seed(test_params.main_seed)
    np.random.seed(test_params.main_seed)

    # get and combine parameters
    (repr_fixed_params, repr_opt_param_domain, repr_opt_param_constraints, make_alg, slurm_args) = select_repr_alg(test_params.repr_alg)
    domain = general_param_domain + repr_opt_param_domain
    constraints = general_param_constraints + repr_opt_param_constraints
    fixed_params = SimpleNamespace(
      **test_params.__dict__,
      **repr_fixed_params.__dict__,
      **pred_params.__dict__,
    )

    # optimize
    logging.info('Running parameter optimization...')
    best_params = run_optimization(args, fixed_params, domain, constraints, slurm_args=slurm_args)

    # test
    logging.info('Running final testing...')
    run_test(args, fixed_params, best_params, test_params.test_seeds, slurm_args=slurm_args)


# run
batch2.run(task_fun=task, main_fun=main)

