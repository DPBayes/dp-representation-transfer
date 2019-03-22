import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms
import matplotlib

from common import ensure_dir_exists, num_ids_from_args, args_from_id


# data types
data_set = 'TCGA_geneexpr_filtered_redistributed'
#data_set = "TCGA_geneexpr_filtered"

repr_dim = 10

seeds = [0, 1, 2]
#seed = 0

# algorithms
algorithms = [
  #'pca',
  #'ae1_64xR_dropout12_Adam',
  #'ae2_32xR_dropout12_Adam',
  #'ae3_32xR_dropout12_Adam',
  'ae1aux_64xR_dropout12',
  'ae2aux_32xR_dropout12',
  'ae3aux_32xR_dropout12',
  #'ae1aux_64xR_dropout24',
  #'ae2aux_32xR_dropout24',
  #'ae3aux_32xR_dropout24',
]
weights = [0.0, .001, .002, .003, .005, .010, .015, .025, .05, .15, .5, (1-.15), (1-.05), (1-.015), (1-.005), 1.0]
optimizer = 'Adam'

drug_acc = True
drugs = range(265)
pv_size = [800]; eps = [1.0, np.inf]
num_cv = 20
ica = True
clipping_only = True

id_suffix = ""
#id_suffix = "-ess" # early stopping secondary (loss)

#fig_name_suffix = "_" + data_set + ("_s" + str(seed)) + id_suffix
fig_name_suffix = "_" + data_set + ("_s" + str(len(seeds)) + "x") + id_suffix
#fig_name_suffix = "_" + data_set

error_bars = True

#tiled = False
#tiled = (3, 2) 
tiled = (1, len(algorithms))

nrows = 2 if drug_acc else 1
fig, axes = plt.subplots(nrows=nrows, ncols=len(algorithms),
                         figsize=(32,20), sharex=True, sharey='row')
axes = axes.reshape((nrows, len(algorithms)))
#fig, axes = plt.subplots(nrows=1, ncols=len(algorithms), figsize=(16,10))

for a, alg in enumerate(algorithms):
  print("alg = %s" % alg)
  tcga_tot_val_loss = np.full((len(weights), len(seeds)), np.nan)
  tcga_dec_val_loss = np.full((len(weights), len(seeds)), np.nan)
  tcga_aux_val_loss = np.full((len(weights), len(seeds)), np.nan)
  tcga_dec_val_mse = np.full((len(weights), len(seeds)), np.nan)
  tcga_aux_val_ce = np.full((len(weights), len(seeds)), np.nan)
  gdsc_dec_mse = np.full((len(weights), len(seeds)), np.nan)
  if drug_acc:
    gdsc_drug_corr_einf = np.full((len(weights), len(seeds)), np.nan)
    gdsc_drug_corr_e1 = np.full((len(weights), len(seeds)), np.nan)
  for w, weight in enumerate(weights):
    print("  weight = %s" % weight)
    for s, seed in enumerate(seeds):
      alg_id = "%s_w%s_%s-s%d%s" % (alg, weight, optimizer, seed, id_suffix)
      #alg_id = "%s_w%s_%s" % (alg, weight, optimizer)
      prefix = "log/%s-%d-%s" % (data_set, repr_dim, alg_id)
      tot_filename = prefix + "-loss.txt"
      tot_val_filename = prefix + "-val_loss.txt"
      dec_filename = prefix + "-decoded_loss.txt"
      dec_val_filename = prefix + "-val_decoded_loss.txt"
      aux_filename = prefix + "-secondary_loss.txt"
      aux_val_filename = prefix + "-val_secondary_loss.txt"
      tot_loss = np.loadtxt(tot_filename)
      tot_val_loss = np.loadtxt(tot_val_filename)
      if tot_val_loss.size < 1:
        continue
      dec_loss = np.loadtxt(dec_filename)
      dec_val_loss = np.loadtxt(dec_val_filename)
      aux_loss = np.loadtxt(aux_filename)
      aux_val_loss = np.loadtxt(aux_val_filename)
      mse_val_filename = "res/progress-encdec-validation-mse-%s-%d-%s.txt" % (data_set, repr_dim, alg_id)
      dec_val_mse = np.loadtxt(mse_val_filename)
      aux_ce_val_filename = "res/progress-aux-validation-ce-%s-%d-%s.txt" % (data_set, repr_dim, alg_id)
      aux_val_ce = np.loadtxt(aux_ce_val_filename)
      tcga_tot_val_loss[w,s] = tot_val_loss[-1]
      tcga_dec_val_loss[w,s] = dec_val_loss[-1]
      tcga_aux_val_loss[w,s] = aux_val_loss[-1]
      tcga_dec_val_mse[w,s] = dec_val_mse[-1]
      tcga_aux_val_ce[w,s] = aux_val_ce[-1]
      filename = "res/private-encdec-rel_mse-%d-%s-%s.txt" % (repr_dim, data_set, alg_id)
      gdsc_dec_mse[w,s] = np.loadtxt(filename) #if os.path.isfile(filename) else np.nan

      # drug pred correlation
      if drug_acc:
        corr = np.empty((len(drugs), num_cv, len(eps)))
        corr[:] = np.nan
        for d, drug in enumerate(drugs):
          prefix = "drugsens_res/drugsens%s-corr-norm2%s-repr-%d-%s-%s" % (
                  ("-ica" if ica else ""), ("-cliponly" if clipping_only else ""),
                  repr_dim, data_set, alg_id)
          
          filename = "%s-%s.npy" % (prefix, drug)
          corr[d,:,:] = np.load(filename)[:,-1,:]

        avg_corr = np.nanmean(corr, axis=0)
        mean = np.nanmean(avg_corr, axis=0)
        #n = np.sum(1-np.isnan(avg_corr), axis=0)
        #quantiles = np.sort(avg_corr, axis=0)
        #yi = np.arange(ny)
        #err = [quantiles[np.floor(.25*(n-1)).astype(int),yi] - mean,
        #      -quantiles[np.ceil(.75*(n-1)).astype(int),yi] + mean]
        gdsc_drug_corr_e1[w,s] = mean[0]
        gdsc_drug_corr_einf[w,s] = mean[1]
  pca_filename = "res/progress-encdec-validation-mse-%s-%d-%s.txt" % (data_set, repr_dim, 'pca')
  pca_tcga_dec_val_mse = np.loadtxt(pca_filename)
  pca_filename = "res/private-encdec-rel_mse-%d-%s.txt" % (repr_dim, 'pca')
  pca_gdsc_dec_mse = np.loadtxt(pca_filename)
  tcga_dec_val_loss_mean = np.mean(tcga_dec_val_loss, axis=1)
  tcga_aux_val_loss_mean = np.mean(tcga_aux_val_loss, axis=1)
  tcga_tot_val_loss_mean = np.mean(tcga_tot_val_loss, axis=1)
  tcga_dec_val_mse_mean = np.mean(tcga_dec_val_mse, axis=1)
  tcga_aux_val_ce_mean = np.mean(tcga_aux_val_ce, axis=1)
  gdsc_dec_mse_mean = np.mean(gdsc_dec_mse, axis=1)
  x = np.arange(len(weights))
  axes[0,a].plot(x, x**0, ':k')
  axes[0,a].plot(x, x**0*.1, ':k')
  if error_bars:
    def yerr(y):
      return [np.mean(y, axis=1) - np.min(y, axis=1),
              np.max(y, axis=1) - np.mean(y, axis=1)]
    capsize=5
    axes[0,a].errorbar(x, tcga_aux_val_loss_mean, yerr(tcga_aux_val_loss), fmt='g', capsize=capsize)
    axes[0,a].errorbar(x, tcga_dec_val_loss_mean, yerr(tcga_dec_val_loss), fmt='b', capsize=capsize)
    axes[0,a].errorbar(x, tcga_tot_val_loss_mean, yerr(tcga_tot_val_loss), fmt='r', capsize=capsize)
    axes[0,a].errorbar(x, tcga_dec_val_mse_mean, yerr(tcga_dec_val_mse), fmt='b', capsize=capsize)
    axes[0,a].errorbar(x, tcga_aux_val_ce_mean, yerr(tcga_aux_val_ce), fmt='g', capsize=capsize)
    axes[0,a].errorbar(x, gdsc_dec_mse_mean, yerr(gdsc_dec_mse), fmt='m', capsize=capsize)
  axes[0,a].plot(x, tcga_dec_val_loss_mean, '-ob', label="tcga dec loss (val)")
  axes[0,a].plot(x, tcga_aux_val_loss_mean, '-og', label="tcga aux loss (val)")
  axes[0,a].plot(x, tcga_tot_val_loss_mean, '-or', label="tcga tot loss (val)")
  axes[0,a].plot(x, x**0*pca_tcga_dec_val_mse, '--b', label="tcga dec mse (val) [PCA]")
  axes[0,a].plot(x, tcga_dec_val_mse_mean, '-.ob', label="tcga dec mse (val)")
  axes[0,a].plot(x, tcga_aux_val_ce_mean, '-.og', label="tcga aux ce (val)")
  axes[0,a].plot(x, x**0*pca_gdsc_dec_mse, '--m', label="gdsc dec mse [PCA]")
  axes[0,a].plot(x, gdsc_dec_mse_mean, '-om', label="gdsc dec mse")
  
  axes[0,a].set_xticks(x)
  axes[0,a].set_xticklabels(weights, rotation=30)
  axes[0,a].set_yscale('log')
  axes[0,a].set_ylim([5e-2, 2e0])
  #axes[0,a].set_yscale('symlog', linthreshy=1e-1)
  if not drug_acc:
    axes[0,a].set_xlabel("aux loss weight")
  if a == 0:
    axes[0,a].set_ylabel("loss")
  if a == 0:
    axes[0,a].legend()
  axes[0,a].set_title(alg)

  if drug_acc:
    axes[1,a].plot(x, gdsc_drug_corr_einf, '-ok', label="pred corr (e=inf)")
    axes[1,a].plot(x, gdsc_drug_corr_e1, '-oy', label="pred corr (e=1)")
    axes[1,a].set_xlabel("aux loss weight")
    if a == 0:
      axes[1,a].set_ylabel("corr")
    if a == 0:
      axes[1,a].legend()

ensure_dir_exists("figs")
figname = "figs/tcga-gdsc-loss-accuracy%s" % (fig_name_suffix)
plt.savefig(figname)
plt.close()
