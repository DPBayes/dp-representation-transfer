import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms
import matplotlib

from common import ensure_dir_exists, num_ids_from_args, args_from_id


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

priv_pairs = [0, 1, 2, 3, 4, 5, 9, 13]

algs = [
  ('rand_proj', "RP", {}),
  ('PCA', "PCA", {}),
  ('VAE', "VAE", {}),
  #('VAE_hyper',{}),
 ]

test_id = ""

n_test_seeds = 9

def np_loadtxt_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.loadtxt(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

def np_load_or(filename, fallback):
  if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    return np.load(filename)
  else:
    print("    Warning: File not found or empty: %s" % (filename))
    return fallback

#average = np.median
#bar_hi = np.max
#bar_lo = np.min

average = np.mean
bar_hi = lambda *args, **kwargs: (
	np.mean(*args, **kwargs) + np.std(*args, **kwargs) / np.sqrt(n_test_seeds))
bar_lo = lambda *args, **kwargs: (
	np.mean(*args, **kwargs) - np.std(*args, **kwargs) / np.sqrt(n_test_seeds))


plot_type = 'bars'
#plot_type = 'all_points'
#plot_type = 'boxplot'

#fig, axes = plt.subplots(nrows=2, ncols=1,
#                         figsize=(8,12), sharex=True, sharey=True)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
main_axes = axes

ylim = (0.55, 0.95)

x0 = np.arange(len(priv_pairs))
for a, (repr_alg, repr_alg_name, style_args) in enumerate(algs):
  x = x0 + (a-1) * 0.2
  acc = np.full((len(priv_pairs), n_test_seeds), np.nan)
  opt_acc = np.full((len(priv_pairs)), np.nan)
  for pv, priv_id in enumerate(priv_pairs):
    priv = cancer_type_pairs[priv_id]
    data_name = (('-'.join(['priv',] + priv)).replace(' ', '_').replace('&', '_'))
    test_name = "%s%s-%s" % (test_id, data_name, repr_alg)
    filename = "res/test_results-%s.txt" % (test_name)
    try:
      acc[pv,:] = np_loadtxt_or(filename, np.array(np.nan))
    except:
      print("Error: Loading '%s' failed: %s" % (filename, sys.exc_info()[0]))

    filename = "param_opt/opt_results-%s.npy" % (test_name)
    opt_results = np_load_or(filename, np.array(np.nan))
    opt_acc[pv] = np.amax(opt_results)

  acc[np.isnan(acc)] = 0.5

  avg_acc = average(acc, axis=1)

  if plot_type == 'bars':
    def yerr(y):
      return [average(y, axis=1) - bar_lo(y, axis=1),
              bar_hi(y, axis=1) - average(y, axis=1)]
    capsize = 5
    main_axes.errorbar(x, avg_acc, yerr(acc), fmt='o', capsize=capsize, label=repr_alg_name, **style_args)
  elif plot_type == 'all_points':
    main_axes.plot(np.repeat(x, n_test_seeds), acc.flatten(), 'o', label=repr_alg_name, **style_args)
  elif plot_type == 'boxplot':
    assert False, "not implemented"
    #main_axes.boxplot(acc, positions=x, whis='range' , label=repr_alg_name, **style_args)
  else:
    assert False, "invalid plot_type"

main_axes.plot(np.tile(x0[np.newaxis, :-1], (2, 1)) + .5,
               np.tile(np.array(ylim)[:,np.newaxis], (1, len(priv_pairs)-1)),
               ':', color='gray', linewidth=1)

#main_axes.set_title("final result with priv")s
main_axes.set_xticks(x0)
#main_axes.set_xticklabels(priv_pairs)
main_axes.set_xticklabels([a+1 for a in x0])
main_axes.tick_params(axis='x', which='both',length=0)
main_axes.set_xlim(-0.5, len(priv_pairs) - 0.5)
main_axes.set_ylim(ylim)
main_axes.set_ylabel("prediction accuracy")
main_axes.set_xlabel("case number")
main_axes.legend()

ensure_dir_exists("figs")
figname = "figs/pred_acc"
#plt.savefig(figname)
plt.tight_layout()
plt.savefig(figname + ".png", format='png', dpi=300)
plt.savefig(figname + ".pdf", format='pdf', dpi=300)
plt.close()
