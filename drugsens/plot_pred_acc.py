# Plots per-drug correlations
#
# Teppo Niinimäki 2017

import numpy as np
import scipy.stats
import csv
import os.path
import sys
import matplotlib.pyplot as plt
from common import ensure_dir_exists


np.random.seed(0)

#inpath = 'drugsens_res/'   # set path for individual files from different drugs and folds
figpath = 'figs/drugsens/'
figname = figpath + "avg-corr-norm2"

#aux_data_set = "TCGA_geneexpr_filtered_redistributed"
aux_data_set = "TCGA_geneexpr_filtered"

orig_data_name = "GDSC_geneexpr_filtered"
redistr_data_name = "GDSC_geneexpr_filtered_redistributed"

dim_reds = [
  ('%s-preselected_10' % (orig_data_name), 'orig_presel_10', {}, False),
  ('%s-preselected_10' % (redistr_data_name), 'redistr_presel_10', {}, False),
]

kifer_dims = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
for dim in kifer_dims:
  #dim_reds.append(('%s-kifer_%d' % (orig_data_name, dim), 'orig_SamAgg_%d' % (dim), {'color':'darkgray'}, True))
  dim_reds.append(('%s-kifer_%d' % (orig_data_name, dim), 'orig_SamAgg 2-20' if dim==2 else None, {'color':'darkgray'}, True))
for dim in kifer_dims:
  #dim_reds.append(('%s-kifer_%d' % (redistr_data_name, dim), 'redistr_SamAgg_%d' % (dim), {'color':'gray'}, True))
  dim_reds.append(('%s-kifer_%d' % (redistr_data_name, dim), 'redistr_SamAgg 2-20' if dim==2 else None, {'color':'gray'}, True))

model_seeds = list(range(9))
#model_seeds = list(range(2))

for alg, alg_name, style in [
            ('rand_proj', "RP", {}),
            ('PCA', "PCA", {}),
            ('VAE', "VAE", {}),
           ]:
  dim_reds.append(('%s-%s-%s' % (redistr_data_name, aux_data_set, alg), alg_name, style, True))

drugids = range(265)
#seeds = range(50)
#seeds = range(20)
seeds = range(10)
#drugids = range(10)
#seeds = range(10)
#drugids = [0]
#seeds = [0]

# Test cases
#n_pv = 800; n_npv = 10 
#n_pv = 875; n_npv = 10
n_pv = 885; n_npv = 0
eps = [1.0, np.inf]
n_test = 100

mcmc = False # use priors instead of fixed values for precision parameter lambda,lambda_0
ica = False
clipping_only = False   # only clip without adding noise


average = np.median
bar_hi = np.max
bar_lo = np.min

plot_type = 'bars'
#plot_type = 'all_points'
#plot_type = 'boxplot'

####################################

ny = len(eps)

colors = ['k','gray','r','b','darkcyan','g','lime','cyan','magenta']

n_files_not_found = 0
last_not_found = None

plt.figure(figsize=(8,6))

x0 = np.arange(ny)
for m, (full_method_id, method, style_args, multiple_seeds) in enumerate(dim_reds):
  x = x0 + m * 0.03
  print("  Method: %s" % (method))

  if multiple_seeds:
    models = ["%s-%d" % (full_method_id, model_seed) for model_seed in model_seeds]
  else:
    models = [full_method_id]
  corr = np.full((len(drugids), len(seeds), ny, len(models)), np.nan)

  # Drugs
  for i, drug in enumerate(drugids):
    print("Drug: %d" % drug)

    for j, model in enumerate(models):
      resname = "%s-pv%dnpv%dtst%d%s%s%s-%d" % (
        model,
        n_pv, n_npv, n_test,
        ("-ica" if ica else ""),
        ("-cliponly" if clipping_only else ""),
        ("-mcmc" if mcmc else "-fixed"),
        drug,
      )
      filename = "drugsens_res/corr-%s.npy" % (resname)
      if os.path.isfile(filename):
        r = np.load(filename)
        
        if r.shape[0] != len(seeds) or r.shape[1] != ny:
          print('Incomplete file: ' + filename)
        
        corr[i,:,:,j] = r
      else:
        #sys.exit('Missing file: '+filename)
        n_files_not_found += 1
        last_not_found = filename

  acc = np.nanmean(corr, axis=1) # average over DP seeds
  acc = np.nanmean(acc, axis=0) # average over drugs
  avg_acc = average(acc, axis=1) # average over model seeds
  #std = np.nanstd(avg_corr, axis=0, ddof=1)
  #n = np.sum(1-np.isnan(avg_corr), axis=0)
  #err = scipy.stats.t.ppf(1-(1-0.95)/2, df=n-1) * std / np.sqrt(n)
  #quantiles = np.sort(avg_corr, axis=0)
  #yi = np.arange(ny)
  #err = [quantiles[np.floor(.25*(n-1)).astype(int),yi] - mean,
  #      -quantiles[np.ceil(.75*(n-1)).astype(int),yi] + mean]
  #print(err)

  if plot_type == 'bars':
    def yerr(y):
      return [average(y, axis=1) - bar_lo(y, axis=1),
              bar_hi(y, axis=1) - average(y, axis=1)]
    capsize = 5
    plt.errorbar(x, avg_acc, yerr(acc), fmt='o', capsize=capsize, label=method, **style_args)
  elif plot_type == 'all_points':
    plt.plot(np.repeat(x, n_test_seeds), acc.flatten(), 'o', label=method, **style_args)
  elif plot_type == 'boxplot':
    assert False, "not implemented"
    #axes[1].boxplot(acc, positions=x, whis='range' , label=repr_alg, **style_args)
  else:
    assert False, "invalid plot_type"

  '''x = np.arange(ny)
  x_pert = x + m * .05
  x_labels = eps
  #x_pert = x + m * 5

  c = colors[m]
  plt.plot(x_pert, mean, 's-', label=method, color=c)
  #plt.errorbar(x_pert, mean, yerr=[std,std], fmt='none', color=c)
  plt.errorbar(x_pert, mean, yerr=err, fmt='none', ecolor=c)
  plt.gca().set_xticks(x)
  plt.gca().set_xticklabels(x_labels)
  plt.xlabel("epsilon")
  plt.legend(loc='lower right')'''

plt.gca().set_xticks(x0)
x_labels = eps
plt.gca().set_xticklabels(x_labels)
plt.xlabel("epsilon")
plt.legend()
  
if n_files_not_found > 0:
  print("Warning: '%s' and %d other files not found." %
      (last_not_found, n_files_not_found-1))

#plt.show()

ensure_dir_exists(figpath)

figname = "%s-npv%d%s%s%s.png" % (figname,
        n_npv,
        ("-ica" if ica else ""),
        ("-cliponly" if clipping_only else ""),
        ("-mcmc" if mcmc else "-fixed"),
)

plt.savefig(figname, format='png', dpi=300, bbox_inches='tight')

