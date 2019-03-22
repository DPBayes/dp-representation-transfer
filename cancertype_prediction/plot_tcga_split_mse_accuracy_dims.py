import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms
import matplotlib

from common import ensure_dir_exists, num_ids_from_args, args_from_id


# data types
#data_set = "TCGA_split_pub_geneexpr"
#aux_data_set = "TCGA_split_pub_geneexpr"
#data_set = "TCGA_split_priv_geneexpr"

'''priv_splits = [
  ("lung squamous cell carcinoma", "head & neck squamous cell carcinoma"),
  ("kidney clear cell carcinoma", "kidney papillary cell carcinoma"),
  ("lung adenocarcinoma", "lung squamous cell carcinoma"),
  ("breast invasive carcinoma", "lung squamous cell carcinoma"),
  ("colon adenocarcinoma", "rectum adenocarcinoma"),
]'''

priv_cancertypes = ("lung squamous cell carcinoma", "head & neck squamous cell carcinoma")
#priv_cancertypes = ("kidney clear cell carcinoma", "kidney papillary cell carcinoma")
#priv_cancertypes = ("lung adenocarcinoma", "lung squamous cell carcinoma")
#priv_cancertypes = ("breast invasive carcinoma", "lung squamous cell carcinoma")
#priv_cancertypes = ("colon adenocarcinoma", "rectum adenocarcinoma")

#repr_dims = [2, 4, 8, 12, 16]
repr_dims = [2, 4, 8, 16]
#repr_dims = [4, 7, 10, 15]

#seeds = [0, 1, 2, 3, 4]
seeds = [0, 1, 2]
#seed = 0

# algorithms
algorithms = [
  ('rand_proj',{'linewidth':3}),
  ('pca',{'linewidth':3}),
  #('vae_torch_gs_e32_d32',{}),
  #('vae_torch_gi_e32_d32',{}),
  #('vae_torch_ps_e32_d32',{}),
  #('vae_torch_pi_e32_d32',{}),
  #('vae_torch_pi_similar',{}),
  ('vae_gs_e32_d32_minmax01',{}),
  #('vae_gi_e32_d32_minmax01',{}),
  ('vae_gs_e32_d32_meanstd01',{}),
  #('vae_gi_e32_d32_meanstd01',{}),
  ('vae_gs_e32_d32_5quantile01',{}),
  #('vae_gi_e32_d32_5quantile01',{}),
 ]

priv = True
pred_accuracy = True
priv_pred_accuracy = True
#pv_size = [800]; eps = [1.0, np.inf]
#ica = False

#scale_fun = "none"; scale_const = 1.0
#scale_fun = "norm_max"; scale_const = 1.01
#scale_fun = "dims_max"; scale_const = 1.0
scale_fun = "norm_avg"; scale_const = 1.0
#scale_fun = "dims_std"; scale_const = 1.0

priv_clip = "norm"; nonpriv_clip = "norm"
#clip = "norm"
#clip = "dims"

epsilon = 1.0


#id_suffix = "-ess" # early stopping secondary (loss)
id_suffix = ""


data_name = '-'.join(priv_cancertypes).replace(' ', '_')

#fig_name_suffix = "_" + data_set + ("_s" + str(len(seeds)) + "x") + id_suffix + ("-ica" if ica else "") + ("-cliponly" if clipping_only else "")
fig_name_suffix = "_" + data_name + ("_s" + str(len(seeds)) + "x") + id_suffix

error_bars = True

recon_bounds = [2e-1, 1.11e0]
priv_recon_bounds = [4.9e-1, 1.22e0]
pred_bounds = [0.50, 1.0]
priv_pred_bounds = pred_bounds

#average = np.mean
average = np.median

#nrows = 2 if pred_accuracy else 1
fig, axes = plt.subplots(nrows=2, ncols=2,
                         figsize=(20,15), sharex=True, sharey=False)
axes = axes.reshape((2, 2))
#fig, axes = plt.subplots(nrows=1, ncols=len(algorithms), figsize=(16,10))

#fig.suptitle(str(len(seeds)) + " repr seeds, " +
#             (", ICA" if ica else "") + 
#             (", clipping only" if clipping_only else ""))
#fig.suptitle(str(len(seeds)) + " repr seeds")
fig.suptitle(priv_cancertypes[0] + " vs. " + priv_cancertypes[1] + ", " + str(len(seeds)) + " repr seeds")

x0 = np.arange(len(repr_dims)) + 1

def np_loadtxt_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.loadtxt(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

x = x0

for a, (alg, style_args) in enumerate(algorithms):
  x = x + 0.02
  print("alg = %s" % alg)
  tcga_dec_val_mse = np.full((len(seeds),len(repr_dims)), np.nan)
  tcga_dec_mse = np.full((len(seeds),len(repr_dims)), np.nan)
  priv_dec_mse = np.full((len(seeds),len(repr_dims)), np.nan)
  if pred_accuracy:
    pred_acc = np.full((len(seeds),len(repr_dims)), np.nan)
    priv_pred_acc = np.full((len(seeds),len(repr_dims)), np.nan)
  for r, repr_dim in enumerate(repr_dims):
    print("  latent dim = %d" % repr_dim)
    for s, seed in enumerate(seeds):
      alg_id = "%s-s%d%s" % (alg, seed, id_suffix)
      mse_val_filename = "res/progress-encdec-validation-mse-%s-%d-%s.txt" % (data_name, repr_dim, alg_id)
      dec_val_mse = np_loadtxt_or(mse_val_filename, np.array(np.nan))
      mse_filename = "res/progress-encdec-mse-%s-%d-%s.txt" % (data_name, repr_dim, alg_id)
      dec_mse = np_loadtxt_or(mse_filename, np.array(np.nan))
      tcga_dec_val_mse[s,r] = dec_val_mse.flatten()[-1]
      tcga_dec_mse[s,r] = dec_mse.flatten()[-1]

      if priv and not np.isnan(tcga_dec_mse[s,r]):
        filename = "res/private-encdec-rel_mse-%d-%s-%s.txt" % (repr_dim, data_name, alg_id)
        priv_dec_mse[s,r] = np_loadtxt_or(filename, np.array(np.nan))

      # drug pred correlation
      if pred_accuracy and not np.isnan(priv_dec_mse[s,r]):
        filename = "res/cancertype-pred-accuracy-%d-%s-%s-s%d-%s-%d-%s%s.txt" % (repr_dim, data_name, alg, seed, scale_fun, scale_const, nonpriv_clip, "-nonpriv")
        pred_acc[s,r] = np_loadtxt_or(filename, np.array(np.nan))
  
      # drug pred correlation
      if priv_pred_accuracy and not np.isnan(priv_dec_mse[s,r]):
        filename = "res/cancertype-pred-accuracy-%d-%s-%s-s%d-%s-%d-%s%s.txt" % (repr_dim, data_name, alg, seed, scale_fun, scale_const, priv_clip, "-e%g" % (epsilon))
        priv_pred_acc[s,r] = np_loadtxt_or(filename, np.array(np.nan))

  # replace nan's by upper bounds
  tcga_dec_val_mse[np.isnan(tcga_dec_val_mse)] = recon_bounds[1]
  tcga_dec_mse[np.isnan(tcga_dec_mse)] = recon_bounds[1]
  if priv:
    priv_dec_mse[np.isnan(priv_dec_mse)] = priv_recon_bounds[1]
  if pred_accuracy:
    pred_acc[np.isnan(pred_acc)] = pred_bounds[0]
  if priv_pred_accuracy:
    priv_pred_acc[np.isnan(priv_pred_acc)] = priv_pred_bounds[0]

  # clipping to y-bounds
  tcga_dec_val_mse = np.clip(tcga_dec_val_mse, recon_bounds[0], recon_bounds[1])
  tcga_dec_mse = np.clip(tcga_dec_mse, recon_bounds[0], recon_bounds[1])
  if priv:
    priv_dec_mse = np.clip(priv_dec_mse, priv_recon_bounds[0], priv_recon_bounds[1])
  if pred_accuracy:
    pred_acc = np.clip(pred_acc, pred_bounds[0], pred_bounds[1])
  if priv_pred_accuracy:
    priv_pred_acc = np.clip(priv_pred_acc, priv_pred_bounds[0], priv_pred_bounds[1])

  # compute averages
  tcga_dec_val_mse_mean = average(tcga_dec_val_mse, axis=0)
  tcga_dec_mse_mean = average(tcga_dec_mse, axis=0)
  if priv:
    priv_dec_mse_mean = average(priv_dec_mse, axis=0)
  if pred_accuracy:
    pred_acc_mean = average(pred_acc, axis=0)
  if priv_pred_accuracy:
    priv_pred_acc_mean = average(priv_pred_acc, axis=0)
  
  #axes[0,a].plot(x, x**0, ':k')
  #axes[0,a].plot(x, x**0*.1, ':k')
  if error_bars:
    def yerr(y):
      return [average(y, axis=0) - np.min(y, axis=0),
              np.max(y, axis=0) - average(y, axis=0)]
    capsize = 5
    axes[0,0].errorbar(x, tcga_dec_val_mse_mean, yerr(tcga_dec_val_mse), capsize=capsize, label=alg, **style_args)
    #axes[0,1].errorbar(x, tcga_dec_mse_mean, yerr(tcga_dec_mse), capsize=capsize, label=alg, **style_args)
    if priv:
      axes[0,1].errorbar(x, priv_dec_mse_mean, yerr(priv_dec_mse), capsize=capsize, label=alg, **style_args)
    if pred_accuracy:
      axes[1,0].errorbar(x, pred_acc_mean, yerr(pred_acc), capsize=capsize, label=alg, **style_args)
    if priv_pred_accuracy:
      axes[1,1].errorbar(x, priv_pred_acc_mean, yerr(priv_pred_acc), capsize=capsize, label=alg, **style_args)
  
  else:
    axes[0,0].plot(x, tcga_dec_val_mse_mean, '-o', label=alg, **style_args)
    #axes[0,1].plot(x, tcga_dec_mse_mean, '-o', label=alg, **style_args)
    if priv:
      axes[0,1].plot(x, priv_dec_mse_mean, '-o', label=alg, **style_args)
    if pred_accuracy:
      axes[1,0].plot(x, pred_acc_mean, '-o', label=alg, **style_args)
    if priv_pred_accuracy:
      axes[1,1].plot(x, priv_pred_acc_mean, '-o', label=alg, **style_args)

for i in [0,1]:
  for j in [0,1]:
    axes[i,j].set_xticks(x0)
    axes[i,j].set_xticklabels(repr_dims)
    axes[i,j].set_xlabel('latent dim')
    axes[i,j].legend()
axes[0,0].set_title("public recon mse (val)")
axes[0,1].set_title("private recon mse")
axes[1,0].set_title("private pred accuracy (nonprivate), scaling = %d*%s, clipping=%s" % (scale_const, scale_fun, nonpriv_clip))
#axes[1,0].set_title("pred accuracy (eps=inf)")
axes[1,1].set_title("private pred accracy (eps=%g), scaling = %d*%s, clipping=%s" % (epsilon, scale_const, scale_fun, priv_clip))
axes[0,0].set_yscale('log')
axes[0,1].set_yscale('log')
axes[1,0].set_yscale('linear')
axes[1,1].set_yscale('linear')
axes[0,0].set_ylim(recon_bounds)
axes[0,1].set_ylim(priv_recon_bounds)
axes[1,0].set_ylim(pred_bounds)
axes[1,1].set_ylim(priv_pred_bounds)
axes[0,0].axhline(1.0, linestyle=':', color='gray')
axes[0,1].axhline(1.0, linestyle=':', color='gray')
#axes[1,0].axhline(1.0, linestyle=':', color='gray')

  #axes[0,a].set_yscale('symlog', linthreshy=1e-1)


ensure_dir_exists("figs")
figname = "figs/tcga-split-mse-corr-dims%s" % (fig_name_suffix)
plt.savefig(figname)
plt.close()
