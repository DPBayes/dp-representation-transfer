import os.path
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
#  ["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"],
#  ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"],
#  ["colon adenocarcinoma", "rectum adenocarcinoma"],
]

priv_val_pairs = [
  (cancer_type_pairs[0], cancer_type_pairs[1]),
  (cancer_type_pairs[0], cancer_type_pairs[2]),
  (cancer_type_pairs[1], cancer_type_pairs[0]),
  (cancer_type_pairs[1], cancer_type_pairs[2]),
  (cancer_type_pairs[2], cancer_type_pairs[0]),
  (cancer_type_pairs[2], cancer_type_pairs[1]),
  (cancer_type_pairs[0], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[0]),
  (cancer_type_pairs[1], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[1]),
  (cancer_type_pairs[2], cancer_type_pairs[3]),
  (cancer_type_pairs[3], cancer_type_pairs[2]),
  (["lung squamous cell carcinoma", "head & neck squamous cell carcinoma"], ["kidney clear cell carcinoma", "kidney papillary cell carcinoma"]),
]

algs = [
  ('rand_proj',{}),
  ('PCA',{}),
  ('VAE',{}),
 ]

test_id = ""

n_test_seeds = 5

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

fig, axes = plt.subplots(nrows=2, ncols=1,
                         figsize=(8,12), sharex=True, sharey=True)


x0 = np.arange(len(priv_val_pairs))
for a, (repr_alg, style_args) in enumerate(algs):
  x = x0 + a * 0.1
  acc = np.full((len(priv_val_pairs), n_test_seeds), np.nan)
  opt_acc = np.full((len(priv_val_pairs)), np.nan)
  for pv, (priv, val) in enumerate(priv_val_pairs):
    data_name = (('-'.join(['priv',] + priv +
                  ['val',] + val))
                  .replace(' ', '_').replace('&', '_'))
    test_name = "%s%s-%s" % (test_id, data_name, repr_alg)
    filename = "res/test_results-%s.txt" % (test_name)
    acc[pv,:] = np_loadtxt_or(filename, np.array(np.nan))

    opt_results = np_load_or("param_opt/opt_results-%s.npy" % (test_name),
                             np.array(np.nan))
    opt_acc[pv] = np.amax(opt_results)

  acc[np.isnan(acc)] = 0.5
  avg_acc = average(acc, axis=1)

  def yerr(y):
    return [average(y, axis=1) - np.min(y, axis=1),
            np.max(y, axis=1) - average(y, axis=1)]
  
  capsize = 5
  axes[1].errorbar(x, avg_acc, yerr(acc), fmt='o', capsize=capsize, label=repr_alg, **style_args)

  axes[0].plot(x, opt_acc, 'o', label=repr_alg, **style_args)

axes[1].set_title("final result with priv")
axes[1].set_xticks(x0)
axes[1].set_ylabel("prediction accuracy")
axes[1].set_xlabel("id of the split to priv, fake-priv and pub")
axes[1].legend()

axes[0].set_title("optimization result with fake-priv")
axes[0].set_ylabel("prediction accuracy")
#axes[0].set_xticks(x0)
#axes[0].set_xlabel("id of the split to priv, fake-priv and pub")
axes[0].legend()

#plt.show()

ensure_dir_exists("figs")
figname = "figs/pred_acc"
plt.savefig(figname)
plt.close()
