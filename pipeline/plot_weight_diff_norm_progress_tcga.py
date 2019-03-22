import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as plt_gridspec
#import matplotlib.transforms as plt.transforms

from common import ensure_dir_exists


# data types
data_sets = [
  'geneMatrix',
]


# the dimension of input and the representation to be learned
#input_dim = None
#input_dim = 2000
#repr_dims = [10, 50, 100, 200, 500, 1000]
#repr_dims = [10, 50, 100, 200]
#input_dim = 500
#repr_dims = [10, 20, 50, 100, 200]
#input_dim = 200
#repr_dims = [10, 20, 50, 100]
#input_dim = 50
#repr_dims = [5, 10, 20, 30]

#input_dims = [50]
#input_dims = [50, 200, 500]
#input_dims = [50, 200, 500, 2000]
#input_dims = [50, 200, 500, 2000, 5000]
input_dims = [50, 200, 500, 2000, 5000, None]
repr_dims = [20]

# algorithms
algorithms = [
  #'pca',
  #'ae0_linear',
]

#for optimizer in ['Adagrad', 'Adam', 'Adamax']:
#  algorithms.append('ae1_2x_' + optimizer)

per_patch = True

n_hidden_layers = 3
hidden_dim_mult = 32
for optimizer in ['Adam']:
#for optimizer in ['Adam', 'Adamax']:
#for optimizer in ['Adam', Adagrad']:
  algorithms.append('ae%d_%sxR_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  #algorithms.append('ae%d_%sxR_dropout12_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  #algorithms.append('ae%d_%sxR_dropout25_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  #algorithms.append('ae%d_%sxR_batchnorm_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))

fig_name_suffix = "_ae%d_%sxR" % (n_hidden_layers, hidden_dim_mult)

#tiled = False
#tiled = (3, 2)
#tiled = (len(input_dims), 2)

figsize = (12.0, 6.0)

#if tiled:
plt.figure(figsize=(len(input_dims)*figsize[0], 4*figsize[1]))
gridspec = plt_gridspec.GridSpec(3, len(input_dims), height_ratios=[2,1,1])

def rolling_window(a, window_len, axis=-1):
  axis = axis % len(a.shape)
  shape = (a.shape[:axis] + (a.shape[axis] - window_len + 1,) +
          a.shape[axis+1:] + (window_len,))
  strides = a.strides + (a.strides[axis],)
  return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_mean(a, window_len, axis=-1):
  #res = np.full(a.shape, np.nan)
  #i = [slice(None)] * a.ndim
  #i[axis] = slice(window_len-1, None)
  #res[i] = np.mean(rolling_window(a, window_len, axis), -1)
  #return res
  return np.mean(rolling_window(a, window_len, axis), -1)

def rolling_std(a, window_len, axis=-1):
  return np.std(rolling_window(a, window_len, axis), -1)

for d, data_set in enumerate(data_sets):
  print("data = %s" % data_set)
  for i, input_dim in enumerate(input_dims):
    print(" input_dim = %s" % input_dim)
    for r, repr_dim in enumerate(repr_dims):
      print(" repr_dim = %d" % repr_dim)
      max_epochs = 0
      algs = []
      for alg_id in algorithms:
        #filename = "res/progress-encdec-mse-%s-%s-%d-%s-perpatch.txt" % (data_set, input_dim, repr_dim, alg_id)
        #rel_mse = np.loadtxt(filename)
        prefix = "log/%s-%s-%d-%s" % (data_set, input_dim, repr_dim, alg_id)
        suffix = "-perpatch" if per_patch else ""
        filename = prefix + "-loss" + suffix + ".txt"
        loss = np.loadtxt(filename)
        filename = prefix + "-weight-diff-norm" + suffix + ".txt"
        weight_diff_norm = np.loadtxt(filename)
        filename = prefix + "-weight-diff-dir-coeff" + suffix + ".txt"
        weight_diff_dir_coeff = np.loadtxt(filename)
        max_epochs = np.maximum(max_epochs, loss.size)
        # find abnormal points
        log_loss_diff = np.log(loss[1:]) - np.log(loss[:-1])
        window_len = 10
        abnormality_dev_treshold = 5
        log_loss_diff_dev = (log_loss_diff[window_len:] -
                            rolling_mean(log_loss_diff, window_len)[:-1])
        
        abnormal = (np.abs(log_loss_diff_dev) >
                    abnormality_dev_treshold * rolling_std(log_loss_diff, window_len)[:-1])
        #abnormal = [i + window_len for i in np.nonzero(abnormal)]
        abnormal = window_len + 1 + np.nonzero(abnormal)[0]
        algs.append((alg_id, loss, weight_diff_norm, weight_diff_dir_coeff, abnormal))

      plt.subplot(gridspec[0, i])
      plt.title("data = %s, input_dim = %s, repr_dim = %d" % (data_set, input_dim, repr_dim))
      for alg in algs:
        alg_id, loss, weight_diff_norm, weight_diff_dir_coeff, abnormal = alg
        print("  alg = %s" % alg_id)
        max = loss.size
        plt.plot(np.arange(0, max), loss, '.-', label=alg_id)
        plt.plot(abnormal, loss[abnormal], 'or')
      #plt.yscale('log')
      plt.yscale('symlog', linthreshy=1e-2)
      plt.xlabel("batch")
      #plt.ylim([1e-1, 1e2])
      plt.xlim([0, max_epochs])
      plt.ylabel("loss (mse)")
      plt.legend()
      
      plt.subplot(gridspec[1, i])
      for alg in algs:
        alg_id, loss, weight_diff_norm, weight_diff_dir_coeff, abnormal = alg
        print("  alg = %s" % alg_id)
        max = weight_diff_norm.size
        plt.plot(np.arange(0, max), weight_diff_norm, '.-', label=alg_id)
        plt.plot(abnormal, weight_diff_norm[abnormal], 'or')
      plt.yscale('log')
      #plt.yscale('symlog', linthreshy=1e-2)
      plt.xlabel("batch")
      #plt.ylim([1e-1, 1e2])
      plt.xlim([0, max_epochs])
      plt.ylabel("weight diff norm (~ gradient norm)")
      #plt.legend()

      plt.subplot(gridspec[2, i])
      for alg in algs:
        alg_id, loss, weight_diff_norm, weight_diff_dir_coeff, abnormal = alg
        print("  alg = %s" % alg_id)
        max = weight_diff_dir_coeff.size
        plt.plot(np.arange(0, max), weight_diff_dir_coeff, '.-', label=alg_id)
        plt.plot(abnormal, weight_diff_dir_coeff[abnormal], 'or')
      #plt.yscale('log')
      plt.yscale('symlog', linthreshy=1e-2)
      plt.xlabel("batch")
      plt.ylim([-1, 1])
      plt.xlim([0, max_epochs])
      plt.ylabel("weight diff dir coeff (~ gradient dir similarity)")
      #plt.legend()
      #if not tiled:
      #  figname = "figs/progress-weight_diff_norm-%s-%s-%d%s" % (data_set, input_dim, repr_dim, fig_name_suffix)
      #  plt.savefig(figname)
      #  plt.close()

#if tiled:
#figname = "figs/progress-mse-tcga-%s%s" % (input_dim, fig_name_suffix)
figname = "figs/progress-weight_diff_norm-tcga%s" % (fig_name_suffix)
with np.errstate(invalid='ignore'):
  plt.savefig(figname)
plt.close()
