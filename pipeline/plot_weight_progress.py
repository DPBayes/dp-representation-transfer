import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pickle

from common import ensure_dir_exists


# data types
data_types = [
#  'circle',
#  'linear',
#  'supersparse',
#  'sparselinear',
#  'onelayer1',
#  'twolayers1',
#  'threelayers1',
#  'twolayers_1-50-50',
#  'twolayers_2-50-50',
  'twolayers_3-50-50',
#  'twolayers_4-50-50',
#  'twolayers_5-50-50',
#  'twolayers_7-50-50',
]

# seeds
seeds = [0]

# algorithms
algorithms = [
#  'pca',
#  'ae_linear',
#  'ae_linear_Adadelta',
#  'ae_linear_initpca',
#  'ae_linear_pretrainpca',
#  'ae1',
#  'ae2',
#  'ae2_dropout',
#  'ae2b_dropout',
#  'ae2c_dropout',
#  'ae2_pretrainpca',
#  'ae3',
#  'vae',
]

#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
#  algorithms.append('ae1_' + optimizer)

for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
  algorithms.append('ae2_' + optimizer)

#for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
#for optimizer in ['SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  algorithms.append('ae_linear_' + optimizer)

#tiled = False
tiled = (len(algorithms), len(data_types)) 

#figsize = (12.0, 9.0)
figsize = (8.0, 6.0)

#print(mpl.rcParams['axes.color_cycle'])

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for d, data_type in enumerate(data_types):
  print("data = %s ..." % data_type)
  max_epochs = 0
  algs = []
  for a, alg_id in enumerate(algorithms):
    print("  alg = %s ..." % alg_id)
    if tiled:
      plt.subplot(tiled[1], tiled[0], d * tiled[0] + a + 1)
    else:
      plt.figure(figsize=figsize)
    plt.title("data = %s, alg = %s" % (data_type, alg_id))
    s = 0
    filename_prefix = "log/%s-%d-%s" % (data_type, seeds[s], alg_id)
    with open(filename_prefix + "-weight-layer-names.p", 'rb') as f:
      layer_names = pickle.load(f)
    with open(filename_prefix + "-weight-shapes.p", 'rb') as f:
      w_shapes = pickle.load(f)
    w = np.loadtxt(filename_prefix + "-weights.txt")
    ax = plt.gca()
    #color_cycle = ax._get_lines.color_cycle
    #color_cycle = iter(cm.rainbow(np.linspace(0, 1, sum([len(s) for s in w_shapes]))))
    #color_cycle = iter(cm.rainbow(np.linspace(0, 1, len(w_shapes))))
    #color_cycle = iter(cm.nipy_spectral(np.linspace(0, 1, len(w_shapes))))
    #color_cycle = iter(cm.gist_rainbow(np.linspace(0, 1, len(w_shapes))))
    color_cycle = iter(cm.Dark2(np.linspace(0, 1, len(w_shapes))))
    #color_cycle = iter(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    #color_cycle = iter(mpl.rcParams['axes.color_cycle'])
    n = 0
    for i, layer in enumerate(w_shapes):
      name = layer_names[i]
      color = next(color_cycle)
      assert len(layer) <= 4
      line_styles = ['-', '--', '-.', ':']
      for j, w_shape in enumerate(layer):
        line_style = line_styles[j]
        s = np.prod(w_shape)
        #plt.plot(np.arange(0, w.shape[0]-1), w[:,n:(n+s)])
        #plt.plot(w[:,n:(n+s)], color=next(color_cycle), label="layer %d, w %d" % (i, j))
        lines = plt.plot(w[:,n:(n+s)], color=color, linestyle=line_style)
        lines[0].set_label("%s [#%d], w %d" % (name, i, j))
        n = n + s
    #plt.plot(w)
    #plt.yscale('log')
    plt.xlabel("epoch")
    plt.ylabel("weight value")

    #handles, labels = plt.gca().get_legend_handles_labels()
    #newLabels, newHandles = [], []
    #for handle, label in zip(handles, labels):
    #  if label not in newLabels:
    #    newLabels.append(label)
    #    newHandles.append(handle)
    #plt.legend(newHandles, newLabels)
    plt.legend()

    if not tiled:
      ensure_dir_exists("figs/weight_progress")
      figname = "figs/weight_progress/%s-%s" % (data_type, alg_id)
      plt.savefig(figname)
      plt.close()

if tiled:
  ensure_dir_exists("figs/weight_progress")
  figname = "figs/weight_progress/all"
  plt.savefig(figname)
  plt.close()
