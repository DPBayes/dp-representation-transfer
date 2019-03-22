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
#input_dims = [50, 200, 500, 2000, 5000, None]
input_dims = [200, 2000, None]
#input_dims = [66, 200, 2000, None]
#input_dims = [66, 200]
#repr_dims = [20]
repr_dims = [10]

# algorithms
algorithms = [
  'pca',
  #'ae0_linear_Adam',
]

#for optimizer in ['Adagrad', 'Adam', 'Adamax']:
#  algorithms.append('ae1_2x_' + optimizer)

n_hidden_layers = 7
hidden_dim_mult = 32
for optimizer in ['Adam']:
#for optimizer in ['Adam', 'Adamax']:
#for optimizer in ['Adam', Adagrad']:
  #algorithms.append('ae1_2x_' + optimizer)
  #algorithms.append('ae1_4xR_' + optimizer)
  #algorithms.append('ae1_4xR_dropout_' + optimizer)
  #algorithms.append('ae1_%sxR_%s' % (hidden_dim_mult, optimizer))
  #algorithms.append('ae1_%sxR_dropout12_%s' % (hidden_dim_mult, optimizer))
  #algorithms.append('ae1_%sxR_dropout25_%s' % (hidden_dim_mult, optimizer))
  #algorithms.append('ae1_%sxR_batchnorm_%s' % (hidden_dim_mult, optimizer))
  #algorithms.append('ae%d_%sxR_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  algorithms.append('ae%d_%sxR_dropout12_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  algorithms.append('ae%d_%sxR_dropout25_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))
  algorithms.append('ae%d_%sxR_batchnorm_%s' % (n_hidden_layers, hidden_dim_mult, optimizer))

task = 'encdec'
#task = 'encdec65'

fig_name_suffix = "_ae%d_%sxR" % (n_hidden_layers, hidden_dim_mult)

#n_hidden_layers = 1
#hidden_dim_mults = [16, 32, 64]
#for optimizer in ['Adam']:
##for optimizer in ['Adam', 'Adamax']:
##for optimizer in ['Adam', Adagrad']:
#  algorithms.append('ae%d_%%sxR_%s' % (hidden_dim_mult, optimizer))
#  algorithms.append('ae%d_%%sxR_dropout12_%s' % (hidden_dim_mult, optimizer))
#  algorithms.append('ae%d_%%sxR_dropout25_%s' % (hidden_dim_mult, optimizer))
#  algorithms.append('ae%d_%%sxR_batchnorm_%s' % (hidden_dim_mult, optimizer))
#fig_name_suffix = "_ae%d" % (n_hidden_layers)

args = (data_sets, input_dims, repr_dims)

#tiled = False
#tiled = (3, 2) 
tiled = (num_ids_from_args((input_dims,)), num_ids_from_args((repr_dims,)))

figsize = (12.0, 9.0) 

relative_to = None
#relative_to = 'pca'

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

#for d, data_set in enumerate(data_sets):
#  print("data = %s" % data_set)
#  for i, input_dim in enumerate(input_dims):
#    print(" input_dim = %s" % input_dim)
#    for r, repr_dim in enumerate(repr_dims):
#      print(" repr_dim = %d" % repr_dim)
for i in range(num_ids_from_args(args)):
  (data_set, input_dim, repr_dim) = args_from_id(i, args)
  print("data = %s, input_dim = %5s, repr_dim = %3d" % (data_set, input_dim, repr_dim))
  if tiled:
    plt.subplot(tiled[1], tiled[0], i + 1)
  else:
    plt.figure(figsize=figsize)
  plt.title("data = %s, input_dim = %s, repr_dim = %d" % (data_set, input_dim, repr_dim))
  max_epochs = 0
  algs = []
  for alg_id in algorithms:
    filename = "res/progress-%s-mse-%s-%s-%d-%s.txt" % (task, data_set, input_dim, repr_dim, alg_id)
    #print(filename)
    rel_mse = np.loadtxt(filename)
    val_filename = "res/progress-%s-validation-mse-%s-%s-%d-%s.txt" % (task, data_set, input_dim, repr_dim, alg_id)
    #print(val_filename)
    val_rel_mse = np.loadtxt(val_filename) if os.path.isfile(val_filename) else None
    max_epochs = np.maximum(max_epochs, rel_mse.size)
    algs.append((alg_id, rel_mse, val_rel_mse))
    if relative_to == alg_id:
      relative_to_value = rel_mse.copy()
  for alg in algs:
    alg_id, rel_mse, val_rel_mse = alg
    print("  alg = %s" % alg_id)
    if relative_to is not None:
      rel_mse -= relative_to_value
    #rel_mse *= np.random.uniform(0.9, 1.1, rel_mse.shape)
    if rel_mse.size == 1:
      last_rel_mse = rel_mse
      best_rel_mse_epoch = 0
      best_rel_mse = rel_mse
      style = 'o'
    else:
      last_rel_mse = rel_mse[-1] 
      best_rel_mse_epoch = len(rel_mse) - np.argmin(rel_mse[::-1]) - 1
      best_rel_mse = rel_mse[best_rel_mse_epoch]
      style = '-'
    max = rel_mse.size
    line, = plt.plot(np.arange(0, max), rel_mse, style, label=alg_id)
    #plt.plot([max-1, max_epochs-1], [last_rel_mse, last_rel_mse], '--',
    #        color=line.get_color())
    rgb = matplotlib.colors.colorConverter.to_rgb(line.get_color())
    light_rgb = tuple([1 - 0.3 * (1-c) for c in rgb])
    plt.plot([best_rel_mse_epoch, max_epochs-1], [best_rel_mse, best_rel_mse], '-',
            color=light_rgb) #color=line.get_color())
    if val_rel_mse is not None:
      line, = plt.plot(np.arange(0, val_rel_mse.size), val_rel_mse, '--',
            color=line.get_color())
            #color=line.get_color(), label=alg_id+" (val)")
    #plt.plot((max_epochs-1) * np.array([1, 1.02, 1.04]), )
    #plt.gca().annotate('foo', xy=(0.2, 0.0), xytext=(-2.0, 0.3), bbox=dict(boxstyle="round", fc="w"))
  #    offset = transforms.ScaledTranslation(dx, dy,
  #  fig.dpi_scale_trans)
  #    y = ax.transData.inverted().transform(last_rel_mse)
  #    y = y + 
  #shadow_transform = ax.transData.inverted().transform()
    #plt.plot((max_epochs-1) * np.array([1, 1.05]), [last_rel_mse])
  plt.yscale('log')
  #plt.yscale('symlog', linthreshy=1e-1)
  plt.xlabel("epoch")
  if relative_to is None:
    #plt.ylim([0, 1e1])
    plt.ylim([1e-1, 2e0])
    plt.ylabel("relative mse")
  else:
    plt.ylabel("relative mse diff from " + relative_to)
  ensure_dir_exists("figs")
  plt.legend()
  if not tiled:
    figname = "figs/%s-progress-mse-%s-%s-%d%s" % (task, data_set, input_dim, repr_dim, fig_name_suffix)
    plt.savefig(figname)
    plt.close()

if tiled:
  #figname = "figs/progress-mse-tcga-%s%s" % (input_dim, fig_name_suffix)
  figname = "figs/%s-progress-mse-tcga%s" % (task, fig_name_suffix)
  plt.savefig(figname)
  plt.close()
