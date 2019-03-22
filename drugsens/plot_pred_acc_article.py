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
  #('%s-preselected_10' % (orig_data_name), '10 preselected (w/o redistr.)', {'color':'gray'}, False),
  ('%s-preselected_10' % (redistr_data_name), '10 selected', {'color':'darkgray'}, False),
  ('%s-kifer_%d' % (orig_data_name, 4), 'SAF (r=%d)' % (4), {'color': 'black'},  True)
]

model_seeds = list(range(9))

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
eps = [1.0, np.inf]; eps_index = 0
n_test = 100

mcmc = False # use priors instead of fixed values for precision parameter lambda,lambda_0
ica = False
clipping_only = False   # only clip without adding noise


#average = np.median
#bar_hi = np.max
#bar_lo = np.min

average = np.mean
bar_hi = lambda *args, **kwargs: (
  np.mean(*args, **kwargs) + np.std(*args, **kwargs) / np.sqrt(len(model_seeds)))
bar_lo = lambda *args, **kwargs: (
  np.mean(*args, **kwargs) - np.std(*args, **kwargs) / np.sqrt(len(model_seeds)))

plot_type = 'bars'
#plot_type = 'all_points'
#plot_type = 'boxplot'

####################################

ny = len(eps)

colors = ['k','gray','r','b','darkcyan','g','lime','cyan','magenta']

n_files_not_found = 0
last_not_found = None

#plt.figure(figsize=(8,6))
#plt.figure(figsize=(3.5,6))
plt.figure(figsize=(2.4,5))

#x0 = np.arange(ny)
x0 = np.arange(1)
for m, (full_method_id, method, style_args, multiple_seeds) in enumerate(dim_reds):
  x = x0 + (m-2) * 0.15
  print("  Method: " + method)

  if multiple_seeds:
    models = ["%s-%d" % (full_method_id, model_seed) for model_seed in model_seeds]
  else:
    models = [full_method_id]
  corr = np.full((len(drugids), len(seeds), ny, len(models)), np.nan)

  # Drugs
  for i, drug in enumerate(drugids):
    #print("Drug: %d" % drug)

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
  acc = acc[eps_index:eps_index+1,:] # select epsilon
  avg_axis = 1
  avg_acc = average(acc, axis=avg_axis) # average over model seeds

  '''corr = corr[:,:,eps_index:eps_index+1,:] # select epsilon
  print(corr.shape)
  acc = np.nanmean(corr, axis=0) # average over drugs
  print(acc.shape)
  if multiple_seeds:
    acc = acc[model_seeds, :, model_seeds] # match model seeds an DP seeds
  else:
    acc = acc[model_seeds, :, 0] # only DP seeds
  print(acc.shape)
  avg_axis = 0
  avg_acc = np.nanmean(acc, axis=avg_axis) # average over DP seeds (and matched model seeds)
  print(avg_acc.shape)
  '''
  

  #style_args = {}
  if plot_type == 'bars':
    def yerr(y):
      return [average(y, axis=avg_axis) - bar_lo(y, axis=avg_axis),
              bar_hi(y, axis=avg_axis) - average(y, axis=avg_axis)]
    capsize = 5
    plt.errorbar(x, avg_acc, yerr(acc), fmt='o', capsize=capsize, label=method, **style_args)
  elif plot_type == 'all_points':
    plt.plot(np.repeat(x, acc.shape[avg_axis]), acc.flatten(), 'o', label=method, **style_args)
  elif plot_type == 'boxplot':
    assert False, "not implemented"
    #axes[1].boxplot(acc, positions=x, whis='range' , label=repr_alg, **style_args)
  else:
    assert False, "invalid plot_type"


plt.gca().set_xticks(x0)
plt.gca().set_xticklabels([" " for a in x0])
plt.gca().tick_params(axis='x', which='both',length=0)
#plt.gca().set_xticks([])
plt.gca().set_xlim(-0.5, 0.5)
plt.gca().set_ylim(0.08, 0.38)
plt.legend()
plt.gca().set_ylabel("prediction accuracy")
plt.gca().set_xlabel("  ")


if n_files_not_found > 0:
  print("Warning: '%s' and %d other files not found." %
      (last_not_found, n_files_not_found-1))

#plt.show()

ensure_dir_exists(figpath)

figname = "%s%s%s%s" % (figname,
        ("-ica" if ica else ""),
        ("-cliponly" if clipping_only else ""),
        ("-mcmc" if mcmc else "-fixed"),
)

plt.tight_layout()

#plt.savefig(figname, format='png', dpi=300, bbox_inches='tight')
plt.savefig(figname + ".png", format='png', dpi=300)
plt.savefig(figname + ".pdf", format='pdf', dpi=300)

