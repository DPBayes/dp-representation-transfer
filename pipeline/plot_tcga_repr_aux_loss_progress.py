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

#seeds = [0, 1, 2, 3, 4]
seed = 0

# algorithms
algorithms = [
  #'pca',
  #'ae1_64xR_dropout12_Adam',
  #'ae2_32xR_dropout12_Adam',
  #'ae3_32xR_dropout12_Adam',
  'ae1aux_64xR_dropout12',
  'ae2aux_32xR_dropout12',
  'ae3aux_32xR_dropout12',
  #'ae1auxnn_64xR_dropout12',
  #'ae2auxnn_32xR_dropout12',
  #'ae3auxnn_32xR_dropout12',
]
weights = [0.0, .001, .002, .003, .005, .010, .015, .025, .05, .15, .5, (1-.15), (1-.05), (1-.015), (1-.005), 1.0]
optimizer = 'Adam'

#id_suffix = ""
id_suffix = "-ess" # early stopping secondary (loss)

fig_name_suffix = "_" + data_set + ("_s" + str(seed)) + id_suffix
#fig_name_suffix = "_" + data_set + "_dropout24"
#fig_name_suffix = "_" + data_set + "_nn"

#tiled = False
#tiled = (3, 2) 
tiled = (1, len(algorithms))

fig, axes = plt.subplots(nrows=len(algorithms), ncols=len(weights),
                          figsize=(40,20), sharex=True, sharey=True)
#fig, axes = plt.subplots(nrows=1, ncols=len(algorithms), figsize=(16,10))

for a, alg_id in enumerate(algorithms):
  print("alg = %s" % alg_id)
  for w, weight in enumerate(weights):
    print("  weight = %s" % weight)
    #prefix = "log/%s-%d-%s_w%s_%s" % (data_set, repr_dim, alg_id, weight, optimizer)
    prefix = "log/%s-%d-%s_w%s_%s-s%d%s" % (data_set, repr_dim, alg_id, weight, optimizer, seed, id_suffix)
    tot_filename = prefix + "-loss.txt"
    tot_val_filename = prefix + "-val_loss.txt"
    dec_filename = prefix + "-decoded_loss.txt"
    dec_val_filename = prefix + "-val_decoded_loss.txt"
    aux_filename = prefix + "-secondary_loss.txt"
    aux_val_filename = prefix + "-val_secondary_loss.txt"
    tot_loss = np.loadtxt(tot_filename)
    tot_val_loss = np.loadtxt(tot_val_filename)
    dec_loss = np.loadtxt(dec_filename)
    dec_val_loss = np.loadtxt(dec_val_filename)
    aux_loss = np.loadtxt(aux_filename)
    aux_val_loss = np.loadtxt(aux_val_filename)
    axes[a,w].plot(dec_loss, '--b', label="dec")
    axes[a,w].plot(dec_val_loss, '-b', label="dec (val)")
    axes[a,w].plot(aux_loss, '--g', label="aux")
    axes[a,w].plot(aux_val_loss, '-g', label="aux (val)")
    axes[a,w].plot(tot_loss, '--r', label="tot")
    axes[a,w].plot(tot_val_loss, '-r', label="tot (val)")
    axes[a,w].set_yscale('log')
    #axes[a].set_yscale('symlog', linthreshy=1e-1)
    if a == len(algorithms)-1:
      axes[a,w].set_xlabel("epoch")
    axes[a,w].set_ylim([5e-2, 2e0])
    if w == 0:
      #axes[a,w].set_ylabel("loss")
      axes[a,w].set_ylabel(alg_id)
    if a == 0 and w == 0:
      axes[a,w].legend()
    if a == 0:
      axes[a,w].set_title("w = %s" % weight)

ensure_dir_exists("figs")
figname = "figs/encded-aux-loss-progress-tcga%s" % (fig_name_suffix)
plt.savefig(figname)
plt.close()
