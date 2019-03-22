import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms
import matplotlib

from common import ensure_dir_exists, num_ids_from_args, args_from_id


# data types
data_sets = [
  'TCGA_geneexpr_filtered_redistributed',
]

repr_dims = [10]

# algorithms
algorithms = [
  #'pca',
  'ae1_64xR_dropout12_Adam',
  'ae2_32xR_dropout12_Adam',
  'ae3_32xR_dropout12_Adam',
]

fig_name_suffix = ""

args = (data_sets, repr_dims)

#tiled = False
#tiled = (3, 2) 
tiled = (1, num_ids_from_args((repr_dims,)))

figsize = (12.0, 9.0) 

relative_to = None
#relative_to = 'pca'

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for i in range(num_ids_from_args(args)):
  (data_set, repr_dim) = args_from_id(i, args)
  print("data = %s, repr_dim = %3d" % (data_set, repr_dim))
  if tiled:
    plt.subplot(tiled[1], tiled[0], i + 1)
  else:
    plt.figure(figsize=figsize)
  plt.title("data = %s, repr_dim = %d" % (data_set, repr_dim))
  max_epochs = 0
  algs = []
  for alg_id in algorithms:
    prefix = "log/%s-%d-%s" % (data_set, repr_dim, alg_id)
    filename = prefix + "-loss.txt"
    val_filename = prefix + "-val_loss.txt"
    loss = np.loadtxt(filename)
    val_loss = np.loadtxt(val_filename) if os.path.isfile(val_filename) else None
    max_epochs = np.maximum(max_epochs, loss.size)
    algs.append((alg_id, loss, val_loss))
    if relative_to == alg_id:
      relative_to_value = loss.copy()
  for alg in algs:
    alg_id, loss, val_loss = alg
    print("  alg = %s" % alg_id)
    if relative_to is not None:
      loss -= relative_to_value
    #loss *= np.random.uniform(0.9, 1.1, loss.shape)
    if loss.size == 1:
      last_loss = loss
      best_loss_epoch = 0
      best_loss = loss
      style = 'o'
    else:
      last_loss = loss[-1] 
      best_loss_epoch = len(loss) - np.argmin(loss[::-1]) - 1
      best_loss = loss[best_loss_epoch]
      style = '-'
    max = loss.size
    line, = plt.plot(np.arange(0, max), loss, style, label=alg_id)
    #plt.plot([max-1, max_epochs-1], [last_loss, last_loss], '--',
    #        color=line.get_color())
    rgb = matplotlib.colors.colorConverter.to_rgb(line.get_color())
    light_rgb = tuple([1 - 0.3 * (1-c) for c in rgb])
    plt.plot([best_loss_epoch, max_epochs-1], [best_loss, best_loss], '-',
            color=light_rgb) #color=line.get_color())
    if val_loss is not None:
      line, = plt.plot(np.arange(0, val_loss.size), val_loss, '--',
            color=line.get_color())
            #color=line.get_color(), label=alg_id+" (val)")
    #plt.plot((max_epochs-1) * np.array([1, 1.02, 1.04]), )
    #plt.gca().annotate('foo', xy=(0.2, 0.0), xytext=(-2.0, 0.3), bbox=dict(boxstyle="round", fc="w"))
  #    offset = transforms.ScaledTranslation(dx, dy,
  #  fig.dpi_scale_trans)
  #    y = ax.transData.inverted().transform(last_loss)
  #    y = y + 
  #shadow_transform = ax.transData.inverted().transform()
    #plt.plot((max_epochs-1) * np.array([1, 1.05]), [last_loss])
  plt.yscale('log')
  #plt.yscale('symlog', linthreshy=1e-1)
  plt.xlabel("epoch")
  if relative_to is None:
    #plt.ylim([0, 1e1])
    plt.ylim([1e-2, 1e0])
    plt.ylabel("loss")
  else:
    plt.ylabel("loss diff from " + relative_to)
  ensure_dir_exists("figs")
  plt.legend()
  #if not tiled:
  #  figname = "figs/loss-progress-%s-%s-%d%s" % (data_set, input_dim, repr_dim, fig_name_suffix)
  #  plt.savefig(figname)
  #  plt.close()

if tiled:
  figname = "figs/encded-loss-progress-tcga%s" % (fig_name_suffix)
  plt.savefig(figname)
  plt.close()
