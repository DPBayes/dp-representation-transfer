import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms

from common import ensure_dir_exists


# data types
#data_types = [
#  'parabola',
#  'u',
#  'circle',
#  'sphere',
#]
data_types = [
#  'linear',
#  'supersparse',
#  'sparselinear',
#  'onelayer1',
#  'twolayers1',
#  'threelayers1',
  'twolayers_1-50-50',
  'twolayers_2-50-50',
  'twolayers_3-50-50',
  'twolayers_4-50-50',
  'twolayers_5-50-50',
  'twolayers_7-50-50',
#  'twolayers_10-50-50',
]

# seeds
seeds = [0]

# algorithms
algorithms = [
  'pca',
  'ae_linear',
#  'ae1',
#  'ae2',
#  'ae2_dropout',
#  'ae2b_dropout',
#  'ae2c_dropout',
#  'ae2_pretrainpca',
#  'ae3',
#  'vae',
]

for optimizer in ['Adam']:
  algorithms.append('ae1_' + optimizer)

#for optimizer in ['Adam']:
for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
  #algorithms.append('ae2_' + optimizer)
  algorithms.append('ae2_2x_' + optimizer)
  #algorithms.append('ae2b_8x_' + optimizer)
  #algorithms.append('ae2_' + optimizer + "_do")
  #algorithms.append('ae2_' + optimizer + "_do_bn")

#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  algorithms.append('ae3_' + optimizer)

# algorithms
#algorithms = [
#  'pca',
#  'ae_linear_SGD',
#  'ae_linear_initpca',
#  'ae_linear_pretrainpca',
#  'ae_linear_Adam',
#  'ae_linear_initpca_Adam',
#  'ae_linear_pretrainpca_Adam',
#]

#for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
#for optimizer in ['SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#for optimizer in ['Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  algorithms.append('ae_linear_' + optimizer)

#tiled = False
tiled = (3, 2) 

figsize = (12.0, 9.0) 

relative_to = None
#relative_to = 'pca'

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for d, data_type in enumerate(data_types):
  print("data = %s" % data_type)
  if tiled:
    plt.subplot(2, 3, d + 1)
  else:
    plt.figure(figsize=figsize)
  plt.title("data = %s" % data_type)
  max_epochs = 0
  algs = []
  for alg_id in algorithms:
    s = 0
    filename = "res/progress-encdec-mse-%s-%d-%s.txt" % (data_type, seeds[s], alg_id)
    rel_mse = np.loadtxt(filename)
    max_epochs = np.maximum(max_epochs, rel_mse.size)
    algs.append((alg_id, rel_mse))
    if relative_to == alg_id:
      relative_to_value = rel_mse.copy()
  for alg in algs:
    alg_id, rel_mse = alg
    print("  alg = %s" % alg_id)
    if relative_to is not None:
      rel_mse -= relative_to_value
    #rel_mse *= np.random.uniform(0.9, 1.1, rel_mse.shape)
    if rel_mse.size == 1:
      last_rel_mse = rel_mse
      style = 'o'
    else:
      last_rel_mse = rel_mse[-1] 
      style = '-'
    max = rel_mse.size
    line, = plt.plot(np.arange(0, max), rel_mse, style, label=alg_id)
    plt.plot([max-1, max_epochs-1], [last_rel_mse, last_rel_mse], '--',
             color=line.get_color())
    #plt.plot((max_epochs-1) * np.array([1, 1.02, 1.04]), )
    #plt.gca().annotate('foo', xy=(0.2, 0.0), xytext=(-2.0, 0.3), bbox=dict(boxstyle="round", fc="w"))
#    offset = transforms.ScaledTranslation(dx, dy,
#  fig.dpi_scale_trans)
#    y = ax.transData.inverted().transform(last_rel_mse)
#    y = y + 
#shadow_transform = ax.transData.inverted().transform()
    #plt.plot((max_epochs-1) * np.array([1, 1.05]), [last_rel_mse])
  #plt.yscale('log')
  plt.yscale('symlog', linthreshy=1e-4)
  plt.xlabel("epoch")
  if relative_to is None:
    plt.ylim([0, 1e2])
    plt.ylabel("relative mse")
  else:
    plt.ylabel("relative mse diff from " + relative_to)
  ensure_dir_exists("figs")
  plt.legend()
  if not tiled:
    figname = "figs/progress-mse-%s" % (data_type)
    plt.savefig(figname)
    plt.close()

if tiled:
  figname = "figs/progress-mse"
  plt.savefig(figname)
  plt.close()
