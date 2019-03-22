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

priv_pairs = [
  cancer_type_pairs[0],
  cancer_type_pairs[1],
  cancer_type_pairs[2],
  cancer_type_pairs[3],
  cancer_type_pairs[4],
  cancer_type_pairs[5],
  #cancer_type_pairs[6],
  #cancer_type_pairs[7],
  #cancer_type_pairs[8],
  cancer_type_pairs[9],
  #cancer_type_pairs[10],
  #cancer_type_pairs[11],
  #cancer_type_pairs[12],
  cancer_type_pairs[13],
  #cancer_type_pairs[14],
  #cancer_type_pairs[15],
]

algs = [
  ('rand_proj',{}),
  ('PCA',{}),
  ('VAE',{}),
  ('VAE_hyper',{}),
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

average = np.median
bar_hi = np.max
bar_lo = np.min

plot_type = 'bars'
#plot_type = 'all_points'
#plot_type = 'boxplot'

fig, axes = plt.subplots(nrows=2, ncols=1,
                         figsize=(8,12), sharex=True, sharey=True)


x0 = np.arange(len(priv_pairs))
for a, (repr_alg, style_args) in enumerate(algs):
  x = x0 + a * 0.1
  acc = np.full((len(priv_pairs), n_test_seeds), np.nan)
  opt_acc = np.full((len(priv_pairs)), np.nan)
  for pv, priv in enumerate(priv_pairs):
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
    axes[1].errorbar(x, avg_acc, yerr(acc), fmt='o', capsize=capsize, label=repr_alg, **style_args)
  elif plot_type == 'all_points':
    axes[1].plot(np.repeat(x, n_test_seeds), acc.flatten(), 'o', label=repr_alg, **style_args)
  elif plot_type == 'boxplot':
    assert False, "not implemented"
    #axes[1].boxplot(acc, positions=x, whis='range' , label=repr_alg, **style_args)
  else:
    assert False, "invalid plot_type"

  axes[0].plot(x, opt_acc, 'o', label=repr_alg, **style_args)

axes[1].set_title("final result with priv")
axes[1].set_xticks(x0)
axes[1].set_ylabel("prediction accuracy")
axes[1].set_xlabel("id of the split to priv and pub")
axes[1].legend()

axes[0].set_title("avg optimization result with different fake-priv datas")
axes[0].set_ylabel("prediction accuracy")
#axes[0].set_xticks(x0)
#axes[0].set_xlabel("id of the split to priv, fake-priv and pub")
axes[0].legend()

#plt.show()

ensure_dir_exists("figs")
figname = "figs/pred_acc"
plt.savefig(figname)
plt.close()
