#!/bin/env python3
# Differentially private Bayesian linear regression
# Arttu Nieminen 2016-2017, Teppo Niinimäki 2017
# University of Helsinki Department of Computer Science
# Helsinki Institute of Information Technology HIIT

# GDSC/drug sensitivity data
# Precision measures: 
# - Spearman's rank correlation coefficient
# - probabilistic concordance index (afterwards weighted average over drugs (wpc-index) should be computed)

import sys
import os
import logging
import numpy as np
from sklearn import linear_model

import batch

# logging configuration
logging.basicConfig(level=logging.INFO)

#aux_data_set = "TCGA_geneexpr_filtered_redistributed"
#aux_data_set = "TCGA_geneexpr_filtered"

orig_data_name = "GDSC_geneexpr_filtered"
orig_data_type = 'rma_gene_expressions'
redistr_data_name = "GDSC_geneexpr_filtered_redistributed"
redistr_data_type = 'redistributed_gene_expressions'

repr_dims = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
#datas = [(redistr_data_name, redistr_data_type)]
datas = [(orig_data_name, orig_data_type), (redistr_data_name, redistr_data_type)]

model_seeds = list(range(9))
#model_seeds = list(range(2))

drugids = range(265)
#seeds = range(50)
#seeds = range(20)
seeds = range(10)
#drugids = range(10)
#seeds = range(10)
#drugids = [0]
#seeds = [0]

# Test cases
#n_pv = 800
#n_npv = 10
n_npv = 0
eps = [1.0, np.inf]
n_test = 100

mcmc = False # use priors instead of fixed values for precision parameter lambda,lambda_0
clipping_only = False   # only clip without adding noise

sparsity = 1.0
lasso_max_iter = 10000

drugid_list_len = 5
drugid_lists = [drugids[i:i + drugid_list_len] for i in range(0, len(drugids), drugid_list_len)]

def select_features(x, y, n_select, sparsity, epsilon, lasso_max_iter=1000, fit_intercept=True):
  # remove missing values
  missing = np.isnan(y)
  x = x[~missing, :]
  y = y[~missing]
  (n_samples, n_features) = x.shape
  assert y.shape == (n_samples,)
  k = int(np.floor(np.sqrt(n_samples)))  # k-part partitioning
  a_supp = linear_model.Lasso(alpha=sparsity/n_samples, fit_intercept=fit_intercept, max_iter=lasso_max_iter)
  v = np.zeros((k, len(n_select), n_features))
  for i in range(k):
    x_i = x[i::k, :]
    y_i = y[i::k]
    a_supp.fit(x_i, y_i)
    assert a_supp.coef_.shape == (n_features,)
    #print(np.sum(a_supp.coef_ > 0))
    for j, d in enumerate(n_select):
      top_features = np.argsort(-np.abs(a_supp.coef_))[:d]
      v[i, j, top_features] = 1
  selected_features = {}
  for j, d in enumerate(n_select):
    lam = 2 * d / (k * (epsilon))
    #print(np.amax(np.mean(v[:,j,:], axis=0)), lam, flush=True)
    g = np.mean(v[:,j,:], axis=0) + np.random.laplace(scale=lam, size=(n_features))
    selected_features[d] = np.argsort(-g)[:d]
  return selected_features


def task(args):
  (d, (gdsc_data_name, gdsc_data_type), drugids) = args
  logging.info("repr_dim = %s", d)
  logging.info("gdsc_data_name = %s", gdsc_data_name)

  import diffpri as dp
  import csv
  import pandas 

  # Import data
  logging.info("Loading gene expressions...")
  geneexpr = pandas.read_hdf("data/%s.h5" % (gdsc_data_name), gdsc_data_type)
  x_full = geneexpr.as_matrix()
  logging.info(" * size = %s x %s" % x_full.shape)
  (n, d_full) = x_full.shape
  
  logging.info("Loading drug sensitivity data...")
  drugres = pandas.read_hdf("data/GDSC_drugres.h5", 'drug_responses')
  y = drugres.as_matrix()
  logging.info(" * size = %s x %s" % y.shape)

  assert x_full.shape[0] == y.shape[0]

  n_pv = n - n_npv - n_test
  pv_max = n_pv

  logging.info("Running the tests...")

  for drugid in drugids:
    logging.info("drugid = %d" % drugid)

    sd = np.nanstd(y[:,drugid], ddof=1)

    S = np.zeros((len(seeds), len(eps), len(model_seeds)), dtype=np.float64)
    R = np.zeros((len(seeds), len(eps), len(model_seeds)), dtype=np.float64)

    for j, e in enumerate(eps):
      logging.info(" epsilon = %s", e)

      repr_eps = e / 2
      pred_eps = e / 2

      if np.isinf(e):
        w_x = np.inf
        w_y = np.inf
      else:
        w_x = np.asscalar(np.loadtxt(
              "drugsens_params/clipping/wx_n%d_d%d_e%s.txt" % (n_pv, d, pred_eps)))
        w_y = np.asscalar(np.loadtxt(
              "drugsens_params/clipping/wy_n%d_d%d_e%s.txt" % (n_pv, d, pred_eps)))
        
      for model_seed in model_seeds:
        logging.info("  model seed = %d" % model_seed)
        np.random.seed(model_seed)

        logging.info("   selecting features...")
        selected_features = select_features(x_full, y[:,drugid], [d], sparsity, repr_eps, lasso_max_iter=lasso_max_iter, fit_intercept=True)

        x = x_full[:, selected_features[d]]
        d = x.shape[1]

        for seed in seeds:
          logging.info("   seed = %d" % seed)

          #logging.info("    preprocessing...")
        
          # Process data
          nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,x_test,y_test,B_x,B_y,n_train,private = dp.processData(x,y,d,n_test,n_pv,n_npv,pv_max,w_x,w_y,drugid,seed)

          if np.isinf(e):
            private = False
          
          if clipping_only:
            private = False
          
          #logging.info("    fitting and evaluating...")

          # Fit model
          if mcmc:
            pred = dp.predictMCMC(n_train,nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,B_x,B_y,pred_eps,x_test,private)
          else:
            pred = dp.predictL(nxx_pv,nxx_npv,nxy_pv,nxy_npv,B_x,B_y,pred_eps,x_test,private)
          
          # Evaluate
          S[seed,j,model_seed] = dp.precision(pred,y_test)
          R[seed,j,model_seed] = dp.pc(pred,y_test,sd)

    # Save results
    for model_seed in model_seeds:
      dim_red = '%s-kifer_%d-%d' % (gdsc_data_name, d, model_seed)
      resname = "%s-pv%dnpv%dtst%d%s%s-%d" % (
        dim_red,
        n_pv, n_npv, n_test,
        ("-cliponly" if clipping_only else ""),
        ("-mcmc" if mcmc else "-fixed"),
        drugid,
      )
      filename = "drugsens_res/corr-%s.npy" % (resname)
      np.save(filename, S[:,:,model_seed])
      logging.info("saved %s" % filename)
      filename = "drugsens_res/wpc-%s.npy" % (resname)
      np.save(filename, R[:,:,model_seed])
      logging.info("saved %s" % filename)


########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(repr_dims, datas, drugid_lists))
#batch.init(task=task, args_ranges=(repr_dims, datas))
batch.main()
