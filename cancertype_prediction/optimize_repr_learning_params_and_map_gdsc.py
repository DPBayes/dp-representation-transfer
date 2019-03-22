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
data_set = "TCGA_geneexpr_filtered"
data_type = 'rnaseq_tpm_rle_log2_gene_expressions'
target_set = "TCGA_cancertype"
target_type = 'cancer_types'

gdsc_data_set = "GDSC_geneexpr_filtered_redistributed"
gdsc_data_type = 'redistributed_gene_expressions'


cancer_type_pairs = [
  ["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"],
  ["bladder urothelial carcinoma", "cervical & endocervical cancer"],
  ["colon adenocarcinoma", "rectum adenocarcinoma"],
  ["stomach adenocarcinoma", "esophageal carcinoma"],
  ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"],
  ["glioblastoma multiforme", "sarcoma"],
  ["adrenocortical cancer", "uveal melanoma"],
  ["testicular germ cell tumor", "uterine carcinosarcoma"],
  ["lung adenocarcinoma", "pancreatic adenocarcinoma"],
  ["ovarian serous cystadenocarcinoma", "uterine corpus endometrioid carcinoma"],
  ["brain lower grade glioma", "pheochromocytoma & paraganglioma"],
  ["skin cutaneous melanoma", "mesothelioma"],
  ["liver hepatocellular carcinoma", "kidney chromophobe"],
  ["breast invasive carcinoma", "prostate adenocarcinoma"],
  ["acute myeloid leukemia", "diffuse large B-cell lymphoma"],
  ["thyroid carcinoma", "cholangiocarcinoma"],
]

## general test parameters

test_params = SimpleNamespace(
  # selection of private and validation_private cancertypes
  param_opt_folds = 2,

  # RNG seeds
  main_seed = 0,
  test_seeds = list(range(9)),
  #test_seeds = [0, 1, 2, 3, 4],
  #test_seeds = [0],

  # representation learning algorithm
  #repr_alg = "VAE",
  #repr_alg = "rand_proj",
  repr_alg = "PCA",

  run_param_opt = False,
  param_opt_continue = False,
  run_learn_and_map = True,
)

## the parameters that will be optimized
general_param_domain = [
  {'name': 'repr_dim', 'type': 'discrete', 'domain': list(range(2, 20+1))},
  ]
general_param_constraints = []

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

#gpyopt_batch_size = 1
#gpyopt_max_iter = 1
gpyopt_batch_size = 1
gpyopt_max_iter = 100
#gpyopt_max_duration = None
#gpyopt_max_duration = datetime.timedelta(minutes=5)
gpyopt_max_duration = datetime.timedelta(hours=120)
gpyopt_deadline = None

# substitute result given to gpyopt in the case of failures
gpyopt_fail_res = 0.5
gpyopt_fail_res_std = 0.02  # To make sure, that gpyopt does not think that the function
                            # (that is being optimized) is deterministic. 
                            # Not sure, if this is really needed.


## other parameters

test_id = ""

## model definitions and model-specific parameters for representation learning
def select_repr_alg(repr_alg):
  if repr_alg == "VAE":
    # fixed parameters
    fixed_params = SimpleNamespace(
      #repr_learn_max_duration = datetime.timedelta(minutes=1),
      repr_learn_max_duration = datetime.timedelta(hours=1),
      repr_learn_validation_split = 0.2,
    )
    # the parameters that will be optimized
    opt_param_domain = [
      {'name': 'learning_rate_log10', 'type': 'continuous', 'domain': (-6,-2)},
      {'name': 'n_hidden_layers', 'type': 'discrete', 'domain': [1, 2, 3]},
      {'name': 'hidden_layer_size_mul_log10', 'type': 'continuous', 'domain': (0,4)},
      ]
    # any constraints for the parameters that will be optimized
    opt_param_constraints = []
    # alg construction
    def make_alg(data_dim, repr_dim, params):
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
        #input_dropout = params.input_dropout,
        #enc_dropout = params.enc_dropout,
        #dec_dropout = params.dec_dropout,
        optimizer = 'Adam',
        optimizer_params = {'lr': 10.0 ** params.learning_rate_log10},
        n_epochs = 2000,
        #n_epochs = 1,
        early_stopping = True,
        reduce_lr_on_plateau = False,
        batch_size = 64)
      return alg
    slurm_args = slurm_gpu_args
  
  elif repr_alg == "PCA":
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
  pred_bounding_slack = 0.01,   # need some slack due to rounding errors

  # DP privacy status and epsilon
  pred_private = True,
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

def run_optimization(args, fixed_params, domain, constraints, batch_size, max_iter, max_duration=None, deadline=None, slurm_args=None):
  logging.info('Starting parameter optimization...')
  import GPyOpt
  ensure_dir_exists("param_opt")

  if max_duration is not None:
    new_dl = datetime.datetime.now() + max_duration
    if deadline is None or new_dl < deadline:
      deadline = new_dl

  # initial parameters and values
  if fixed_params.param_opt_continue:
    logging.info('Loading earlier params and results...')
    all_params = np.load("param_opt/opt_params-%s.npy" % (fixed_params.test_name))
    all_results = np.load("param_opt/opt_results-%s.npy" % (fixed_params.test_name))
    opt_seeds = range(len(all_params))
  else:
    logging.info('Selecting initial parameters...')
    initial_design_type = 'random'
    initial_design_numdata = batch_size
    space = GPyOpt.core.task.space.Design_space(domain, constraints)
    opt_params = GPyOpt.experiment_design.initial_design(initial_design_type, space, initial_design_numdata)
    logging.info('Running...')
    opt_seeds = range(len(opt_params))
    task_params = [get_params(p, domain) for p in opt_params]
    results = run_optimization_batch(args, fixed_params, task_params, opt_seeds, slurm_args)
    all_params = opt_params
    all_results = results

  for i in range(max_iter):
    #print(np.hstack((all_params, all_results)), flush=True)
    logging.info('Best result this far: %g', np.amax(all_results))
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

    opt_params = bo.suggest_next_locations()
    next_seed = max(opt_seeds) + 1
    opt_seeds = range(next_seed, next_seed + len(opt_params))
    logging.info('Running...')
    task_params = [get_params(p, domain) for p in opt_params]
    results = run_optimization_batch(args, fixed_params, task_params, opt_seeds, slurm_args)
    all_params = np.vstack((all_params, opt_params))
    all_results = np.vstack((all_results, results))
    np.save("param_opt/opt_params-%s.npy" % (fixed_params.test_name), all_params)
    np.save("param_opt/opt_results-%s.npy" % (fixed_params.test_name), all_results)

    if datetime.datetime.now() >= deadline:
      logging.info('Gpyopt iteration %d: Time based stopping' % (i))
      break

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
  assert len(task_params) == gpyopt_batch_size

  res = np.zeros((len(task_params), fixed_params.param_opt_folds))
  for fold in range(fixed_params.param_opt_folds):
    val_cancertype_pairs = [ctp for (i, ctp) in enumerate(cancer_type_pairs)
                            if i % fixed_params.param_opt_folds == fold]
    learn_cancertype_pairs = [ctp for (i, ctp) in enumerate(cancer_type_pairs)
                              if i % fixed_params.param_opt_folds != fold]
    assert (len(val_cancertype_pairs) + len(learn_cancertype_pairs) ==
            len(cancer_type_pairs))
    
    common_params = SimpleNamespace(
      **fixed_params.__dict__,
      priv_cancertype_pairs=val_cancertype_pairs,
      pub_cancertypes=sum(learn_cancertype_pairs, []),
      task_type='paramopt',
    )
    args.wait = True
    batch2.run_tasks(args, common_params, task_params, slurm_args=slurm_args,
      params_file=("run_parameters/batch-%s.pkl" % (fixed_params.test_name)))
    # get results
    for param_id in param_ids:
      full_model_id = "%s-%s" % (fixed_params.test_name, param_id)
      filename = "param_opt/opt_result-%s.txt" % (full_model_id)
      try:
        res[param_id, fold] = np.loadtxt(filename)
        import os
        os.remove(filename)
      except:
        res[param_id, fold] = gpyopt_fail_res + np.random.randn() * gpyopt_fail_res_std
        logging.info('Warning, could not load "%s"' % filename)
  return np.mean(res, axis=1, keepdims=True)


def run_learning_and_mapping(args, fixed_params, task_param, seeds, slurm_args=None):
  logging.info('Running final tests with...')
  import copy
  task_params = [copy.copy(task_param) for s in seeds]
  param_ids = range(len(task_params))
  for param, seed, param_id in zip(task_params, seeds, param_ids):
    param.param_id = param_id
    param.seed = seed
  ensure_dir_exists("run_parameters")

  common_params = SimpleNamespace(
    **fixed_params.__dict__,
    priv_cancertype_pairs=[],
    pub_cancertypes=sum(cancer_type_pairs, []),
    task_type='learn_and_map',
  )
  args.wait = True
  batch2.run_tasks(args, common_params, task_params, slurm_args=slurm_args,
    params_file=("run_parameters/batch-%s.pkl" % (fixed_params.test_name)))


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
  
  logging.info(" * private cancertype pairs: %s" % params.priv_cancertype_pairs)
  logging.info(" * public cancertypes: %s" % params.pub_cancertypes)

  priv_cancertypes = sum(params.priv_cancertype_pairs, [])
  priv = cancer_type.isin(priv_cancertypes)
  pub = cancer_type.isin(params.pub_cancertypes)

  logging.info(" * %d private samples, %d public samples (of %d total)" %
              (sum(priv), sum(pub), priv.size))

  from common import categorical_to_binary

  x_pub = gene_expr[pub].as_matrix()
  y_pub = cancer_type[pub].cat.codes.as_matrix()

  seed = int(params.seed)
  # init rng  
  np.random.seed(seed)
  import torch
  torch.manual_seed(seed)
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    torch.cuda.manual_seed(seed)

  full_model_id = "%s-%s" % (common_params.test_name, task_params.param_id)

  logging.info("Representation learning...")
  repr_alg = learn_repr(x_pub, y_pub, params, full_model_id)
  x_pub_repr = map_repr(x_pub, repr_alg, params, full_model_id)

  if params.task_type == 'paramopt':
    acc = np.zeros(len(params.priv_cancertype_pairs))
    for p, priv_cancertype_pair in enumerate(params.priv_cancertype_pairs):
      logging.info("Prediction with private cancertypes %s..." % priv_cancertype_pair)
      priv = cancer_type.isin(priv_cancertype_pair)
      x_priv = gene_expr[priv].as_matrix()
      y_priv = cancer_type[priv].cat.codes.as_matrix()
      x_priv_repr = map_repr(x_priv, repr_alg, params, full_model_id)
      acc[p] = predict(x_priv_repr, y_priv, x_pub_repr, params, full_model_id)

    avg_acc = np.mean(acc)
    logging.info("Total average prediction accuracy: %.6f" % avg_acc)
    
    logging.info("Writing results to disk...")
    filename = "param_opt/opt_result-%s.txt" % (full_model_id)
    logging.info(" * filename: %s", filename)
    with open(filename, 'w', encoding='utf-8') as f:
      f.write("%.6f\n" % avg_acc)

  elif params.task_type == 'learn_and_map':
    gdsc_gene_expr = load_gdsc_data()
    x_gdsc = gdsc_gene_expr.as_matrix()
    x_gdsc_repr = map_repr(x_gdsc, repr_alg, params, full_model_id)
    
    logging.info("Saving the representation...")
    ensure_dir_exists("data_repr")
    np.savetxt("data_repr/%s-%s.csv" % (gdsc_data_set, full_model_id),
              x_gdsc_repr, delimiter=',')
  else:
    assert False, "invalid task type"


def load_data():
  import pandas
  logging.info("Reading data...")
  gene_expr = pandas.read_hdf("data/%s.h5" % (data_set), data_type)
  logging.info(" * gene expression shape: %d x %d" % gene_expr.shape)

  logging.info("Loading cancer types...")
  cancer_type = pandas.read_hdf("data/%s.h5" %  (target_set), target_type)
  logging.info(" * cancer type samples: %d" % cancer_type.shape)
  #assert np.array_equal(gene_expr.index, cancer_type.index)

  logging.info("Taking only common samples...")
  common_samples = gene_expr.index.intersection(cancer_type.index)
  gene_expr = gene_expr.loc[common_samples]
  cancer_type = cancer_type.loc[common_samples]
  logging.info(" * number of common samples: %d" % common_samples.size)

  return (gene_expr, cancer_type)


def load_gdsc_data():
  import pandas
  logging.info("Reading GDSC data...")
  gene_expr = pandas.read_hdf("data/%s.h5" % (gdsc_data_set), gdsc_data_type)
  logging.info(" * gene expression shape: %d x %d" % gene_expr.shape)

  return gene_expr


##################################
#   representation learning
#################################

def learn_repr(x, y, params, full_model_id):

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

  return alg


##################################
#   representation mapping
#################################

def map_repr(x, alg, params, full_model_id):

  # get the representation
  logging.info("Making the representation of data...")
  x_repr = alg.encode(x)
  logging.info(" * representation shape: %d x %d" % x_repr.shape)

  # test to predict the data itself
  x_pred = alg.decode(x_repr)
  rel_mse = relative_mean_squared_error(x, x_pred)
  logging.info(" * reconstruct the data: rel_mse = %g", rel_mse)

  return x_repr


##################################
#  prediction
#################################

def predict(x, y, x_pub, params, full_model_id):

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

    logging.debug("Bounding the data to 1-sphere...")
    if params.pred_scale_fun == "norm_max":
      logging.debug(" * scale by max norm")
      scale_factor = np.amax(np.linalg.norm(x_pub, axis=1))
    elif params.pred_scale_fun == "dims_max":
      logging.debug(" * scale each dimension by max absolute value")
      scale_factor = np.amax(np.abs(x_pub), axis=0)
    elif params.pred_scale_fun == "norm_avg":
      logging.debug(" * scale by average norm")
      scale_factor = np.mean(np.linalg.norm(x_pub, axis=1))
    elif params.pred_scale_fun == "dims_std":
      logging.debug(" * scale each dimension by standard deviation")
      scale_factor = np.std(x_pub, axis=0)
    elif params.pred_scale_fun == "none":
      scale_factor = 1.0
    else:
      assert False

    x_train /= scale_factor * params.pred_scale_const
    x_test /= scale_factor * params.pred_scale_const
    if params.pred_clip == "norm":
      logging.debug(" * clip norms to max 1")
      x_train /= np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True) * (1 + params.pred_bounding_slack), 1)
      x_test /= np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True) * (1 + params.pred_bounding_slack),1)
    elif params.pred_clip == "dims":
      assert False, "not implemented"
    elif params.pred_clip == "none":
      logging.debug(" * no clipping -> no bounding")
      assert params.pred_private == False #or np.isinf(epsilon)
    else:
      assert False

    # fit
    logging.debug("Fitting a model...")
    if params.pred_private:
      logging.debug(" * DP logistic regression: epsilon=%g, alpha=%g", params.pred_epsilon, params.pred_regularizer_strength)
      from models.logistic_regression import DPLogisticRegression
      model = DPLogisticRegression().init(x.shape[1], classes=np.unique(y),
                                          alpha=params.pred_regularizer_strength, epsilon=params.pred_epsilon)
    else:
      logging.debug(" * logistic regression: alpha=%g", params.pred_regularizer_strength)
      from sklearn.linear_model import LogisticRegression
      model = LogisticRegression(C=1/params.pred_regularizer_strength)
    
    model.fit(x_train, y_train)
    #print(model.predict(x_test))

    # compute mean accuracy on test set
    logging.debug("Testing the model...")
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
    data_name = data_set
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
    if fixed_params.run_param_opt:
      logging.info('Running parameter optimization...')
      best_params = run_optimization(args, fixed_params, domain, constraints, gpyopt_batch_size, gpyopt_max_iter, max_duration=gpyopt_max_duration, deadline=gpyopt_deadline, slurm_args=slurm_args)
    else:
      filename = "param_opt/paramopt_best_params-%s.txt" % (fixed_params.test_name)
      logging.info("Loading best params from '%s'" % filename)
      with open(filename, 'rb') as f:
        best_params = pickle.load(f)

    # test
    if fixed_params.run_learn_and_map:
      logging.info('Running final learning and mapping...')
      run_learning_and_mapping(args, fixed_params, best_params, test_params.test_seeds, slurm_args=slurm_args)


# run
batch2.run(task_fun=task, main_fun=main)

