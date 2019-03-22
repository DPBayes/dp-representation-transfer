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

import batch

# logging configuration
logging.basicConfig(level=logging.INFO)

#aux_data_set = "TCGA_geneexpr_filtered_redistributed"
aux_data_set = "TCGA_geneexpr_filtered"

orig_data_name = "GDSC_geneexpr_filtered"
redistr_data_name = "GDSC_geneexpr_filtered_redistributed"

dim_reds = [
  '%s-preselected_10' % (orig_data_name),
  '%s-preselected_10' % (redistr_data_name),
]

model_seeds = list(range(9))

test_id = ""

for alg in [
            '%s-rand_proj' % (aux_data_set),
            '%s-PCA' % (aux_data_set),
            '%s-VAE' % (aux_data_set),
           ]:
  for model_seed in model_seeds:
    full_model_id = "%s%s-%d" % (test_id, alg, model_seed)
    dim_reds.append('%s-%s' % (redistr_data_name, full_model_id))

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
ica = False
clipping_only = False   # only clip without adding noise

def task(args):
  (dim_red,) = args
  logging.info("dim_red = %s", dim_red)

  import diffpri as dp
  import csv
  import pandas 

  # Import data
  logging.info("Loading representation...")
  filename = "%s.csv" % (dim_red)
  f = open("data_repr/" + filename, 'rt')
  reader = csv.reader(f,delimiter=',')
  x = np.array(list(reader)).astype(float)
  f.close()
  logging.info(" * size = %s x %s" % x.shape)
  (n, d) = x.shape
  
  logging.info("Loading drug sensitivity data...")
  drugres = pandas.read_hdf("data/GDSC_drugres.h5", 'drug_responses')
  y = drugres.as_matrix()
  logging.info(" * size = %s x %s" % y.shape)

  assert x.shape[0] == y.shape[0]

  n_pv = n - n_npv - n_test
  pv_max = n_pv

  if ica:
    logging.info("Running FastICA...")
    from sklearn.decomposition import FastICA
    x = FastICA(max_iter=2000).fit_transform(x)

  logging.info("Running the tests...")

  for drugid in drugids:
    logging.info("drugid = %d" % drugid)
    #S = np.zeros((len(seeds),len(pv_size),len(eps)),dtype=np.float64)
    #R = np.zeros((len(seeds),len(pv_size),len(eps)),dtype=np.float64)
    S = np.zeros((len(seeds), len(eps)), dtype=np.float64)
    R = np.zeros((len(seeds), len(eps)), dtype=np.float64)
    for seed in seeds:
      logging.info("seed = %d" % seed)
      sd = np.nanstd(y[:,drugid], ddof=1)

      for j, e in enumerate(eps):
        if np.isinf(e):
          w_x = np.inf
          w_y = np.inf
        else:
          w_x = np.asscalar(np.loadtxt(
                "drugsens_params/clipping/wx_n%d_d%d_e%s.txt" % (n_pv, d, e)))
          w_y = np.asscalar(np.loadtxt(
                "drugsens_params/clipping/wy_n%d_d%d_e%s.txt" % (n_pv, d, e)))
        
        # Process data
        nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,x_test,y_test,B_x,B_y,n_train,private = dp.processData(x,y,d,n_test,n_pv,n_npv,pv_max,w_x,w_y,drugid,seed)

        if np.isinf(e):
          private = False
        
        if clipping_only:
          private = False
        
        # Fit model
        if mcmc:
          pred = dp.predictMCMC(n_train,nxx_pv,nxx_npv,nxy_pv,nxy_npv,nyy_pv,nyy_npv,B_x,B_y,e,x_test,private)
        else:
          pred = dp.predictL(nxx_pv,nxx_npv,nxy_pv,nxy_npv,B_x,B_y,e,x_test,private)
        
        # Evaluate
        S[seed,j] = dp.precision(pred,y_test)
        R[seed,j] = dp.pc(pred,y_test,sd)

    # Save results
    resname = "%s-pv%dnpv%dtst%d%s%s%s-%d" % (
      dim_red,
      n_pv, n_npv, n_test,
      ("-ica" if ica else ""),
      ("-cliponly" if clipping_only else ""),
      ("-mcmc" if mcmc else "-fixed"),
      drugid,
    )
    filename = "drugsens_res/corr-%s.npy" % (resname)
    np.save(filename, S)
    logging.info("saved %s" % filename)
    filename = "drugsens_res/wpc-%s.npy" % (resname)
    np.save(filename, R)
    logging.info("saved %s" % filename)


########## MAIN ##########

# init and run
#batch.init(task=task, args_ranges=(dim_reds, drugids, seeds))
#batch.init(task=task, args_ranges=(dim_reds, drugids))
batch.init(task=task, args_ranges=(dim_reds, ))
batch.main()
