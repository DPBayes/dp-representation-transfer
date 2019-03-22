import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms
import matplotlib

from common import ensure_dir_exists, num_ids_from_args, args_from_id


# data types
data_sets = [
  'geneMatrix',
]


# the dimension of input and the representation to be learned
#input_dims = [50]
#input_dims = [50, 200, 500]
#input_dims = [50, 200, 500, 2000]
#input_dims = [50, 200, 500, 2000, 5000]
#input_dims = [50, 200, 500, 2000, 5000, None]
input_dims = [200, 2000, None]
#input_dims = [66, 200, 2000, None]
#repr_dims = [20]
repr_dims = [10]

n_hidden_layerss = [1, 2, 3]
hidden_dim_mults = [16, 32, 64]
optimizers = ['Adam']

# algorithms
algorithms = [
  ('pca', 'black')
  #('ae0_linear_Adam', 'black')
]
'''for optimizer in optimizers:
  for n_hidden_layers in n_hidden_layerss:
    for hidden_dim_mult in hidden_dim_mults:
      algorithms.append(('ae%d_%sxR_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'red'))
      algorithms.append(('ae%d_%sxR_dropout12_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'green'))
      algorithms.append(('ae%d_%sxR_dropout25_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'darkgreen'))
      algorithms.append(('ae%d_%sxR_batchnorm_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'blue'))'''

for optimizer in ['Adam']:
  for n_hidden_layers in [3, 4, 5, 6, 7, 8]:
    for hidden_dim_mult in [8, 16, 32]:
      algorithms.append(('ae%db_%sxR_dropout12_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'green'))
      algorithms.append(('ae%db_%sxR_dropout25_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'darkgreen'))
      algorithms.append(('ae%db_%sxR_batchnorm_%s' % (n_hidden_layers, hidden_dim_mult, optimizer), 'blue'))

#def get_algorithms(n_hidden_layers, hidden_dim_mult):
#  algs = [
#    'pca',
#  ]
#  for optimizer in optimizers:
#    algs.append('ae%d_%sxR_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
#    algs.append('ae%d_%sxR_dropout12_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
#    algs.append('ae%d_%sxR_dropout25_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
#    algs.append('ae%d_%sxR_batchnorm_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
#  return algs

task = 'encdec'
#task = 'encdec65'

fig_name_suffix = "_morelayers"

args = (data_sets, input_dims, repr_dims)

#tiled = False
#tiled = (3, 2) 
tiled = (num_ids_from_args((input_dims,)), num_ids_from_args((repr_dims,)))

figsize = (12.0, 9.0) 

relative_to = None
#relative_to = 'pca'

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for i in range(num_ids_from_args(args)):
  (data_set, input_dim, repr_dim) = args_from_id(i, args)
  print("data = %s, input_dim = %5s, repr_dim = %3d" % (data_set, input_dim, repr_dim))
  if tiled:
    plt.subplot(tiled[1], tiled[0], i + 1)
  else:
    plt.figure(figsize=figsize)
  plt.title("data = %s, input_dim = %s, repr_dim = %d" % (data_set, input_dim, repr_dim))
  algs = []
  for (alg_id, color) in algorithms:
    filename = "res/progress-%s-mse-%s-%s-%d-%s.txt" % (task, data_set, input_dim, repr_dim, alg_id)
    #print(filename)
    rel_mse = (np.loadtxt(filename)
              if os.path.isfile(filename) and os.path.getsize(filename) > 0
              else None)
    #if rel_mse is None:
    #  continue
    val_filename = "res/progress-%s-validation-mse-%s-%s-%d-%s.txt" % (task, data_set, input_dim, repr_dim, alg_id)
    #print(val_filename)
    val_rel_mse = (np.loadtxt(val_filename)
                  if os.path.isfile(val_filename) and os.path.getsize(val_filename) > 0
                  else None)
    algs.append((alg_id, rel_mse, val_rel_mse, color))
    if relative_to == alg_id:
      relative_to_value = rel_mse.copy()
  best_rel_mses = []
  best_val_rel_mses = []
  for a, alg in enumerate(algs):
    alg_id, rel_mse, val_rel_mse, color = alg
    print("  alg = %s" % alg_id)
    if rel_mse is None:
      best_rel_mse = np.nan
      best_val_rel_mse = np.nan
    else:
      if relative_to is not None:
        rel_mse -= relative_to_value
      best_rel_mse = np.nanmin(rel_mse)
      best_val_rel_mse = np.nanmin(val_rel_mse)
    best_rel_mses.append(best_rel_mse)
    best_val_rel_mses.append(best_val_rel_mse)
  alg_ids = [alg_id for (alg_id,_,_,_) in algs]
  colors = [color for (_,_,_,color) in algs]
  plt.barh(np.arange(len(algs)), best_val_rel_mses, tick_label=alg_ids, color=colors)
  plt.barh(np.arange(len(algs)), best_rel_mses, color=[0.9, 0.9, 0.9, 0.5])
  #b = np.argmin(best_val_rel_mses)
  best = np.nanmin(best_val_rel_mses)
  plt.plot([best, best], [0, len(algs)], '-k')
  #plt.yscale('log')
  #plt.yscale('symlog', linthreshy=1e-1)
  if relative_to is None:
    plt.xlim([0, 5e-1])
    #plt.ylim([1e-1, 2e0])
    plt.xlabel("relative mse")
  else:
    plt.xlabel("relative mse diff from " + relative_to)
  ensure_dir_exists("figs")

if tiled:
  figname = "figs/%s-mse-tcga%s" % (task, fig_name_suffix)
  plt.savefig(figname)
  plt.close()
