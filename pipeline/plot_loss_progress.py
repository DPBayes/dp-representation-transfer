import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

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
   'vae_test',
]

# seeds
seeds = [0, 1, 2, 3, 4, 5]

# algorithms
algorithms = [
#  'pca',
#  'ae_linear',
#  'ae_linear_Adadelta',
#  'ae_linear_initpca',
#  'ae_linear_pretrainpca',
#  'ae_linear_initpca_Adam',
#  'ae_linear_pretrainpca_Adam',
#  'ae1',
#  'ae2',
#  'ae2_dropout',
#  'ae2b_dropout',
#  'ae2c_dropout',
#  'ae2_pretrainpca',
#  'ae3',
  'vae',
]

# autoencoders
#for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
#for optimizer in ['SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#for optimizer in ['Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  algorithms.append('ae_linear_' + optimizer)

# encoders
#for optimizer in ['Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  algorithms.append('e_' + optimizer)
#for lr in [0.1, 0.01, 0.001]:
#  algorithms.append('e_SGD_%g' % lr)

#tiled = False
tiled = (3, 2) 

figsize = (12.0, 9.0)

noise_log_dev = np.log1p(0.01)

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

#for d, data_type in enumerate(data_types):
data_type = data_types[0]
for s, seed in enumerate(seeds):
  #print("data = %s" % data_type)
  if tiled:
    #plt.subplot(2, 3, d + 1)
    plt.subplot(2, 3, s + 1)
  else:
    plt.figure(figsize=figsize)
  plt.title("data = %s" % data_type)
  max_epochs = 0
  algs = []
  for alg_id in algorithms:
    #s = 0
    prefix = "log/%s-%d-%s" % (data_type, seeds[s], alg_id)
    filename = prefix + "-loss.txt"
    e_filename = prefix + "-encoded_loss.txt"
    d_filename = prefix + "-decoded_loss.txt"
    loss = np.loadtxt(filename)
    e_loss = np.loadtxt(e_filename) if os.path.isfile(e_filename) else None
    d_loss = np.loadtxt(d_filename) if os.path.isfile(d_filename) else None
    max_epochs = np.maximum(max_epochs, loss.size)
    algs.append((alg_id, loss, e_loss, d_loss))
  for alg in algs:
    alg_id, loss, e_loss, d_loss = alg
    print("  alg = %s" % alg_id)
    #loss *= np.exp(np.random.uniform(-noise_log_dev, noise_log_dev, loss.shape))
    loss += 5
    if loss.size == 1:
      last_loss = loss
      style = 'o'
    else:
      last_loss = loss[-1]
      style = '-'
    max = loss.size
    line, = plt.plot(np.arange(1, max+1), loss, style, label=alg_id)
    #plt.plot([max, max_epochs], [last_loss, last_loss], '--',
    #         color=line.get_color())
    if e_loss is not None:
      plt.plot(np.arange(1, e_loss.size+1), e_loss, '--', color=line.get_color())
    if d_loss is not None:
      plt.plot(np.arange(1, d_loss.size+1), d_loss, '-.', color=line.get_color())
    #plt.plot((max_epochs-1) * np.array([1, 1.02, 1.04]), )
    #plt.gca().annotate('foo', xy=(0.2, 0.0), xytext=(-2.0, 0.3), bbox=dict(boxstyle="round", fc="w"))
#    offset = transforms.ScaledTranslation(dx, dy,
#  fig.dpi_scale_trans)
#    y = ax.transData.inverted().transform(last_loss)
#    y = y + 
#shadow_transform = ax.transData.inverted().transform()
    #plt.plot((max_epochs-1) * np.array([1, 1.05]), [last_loss])
  #plt.yscale('log')
  plt.yscale('symlog', linthreshy=1e-6)
  #plt.yscale('symlog', linthreshy=10)
  plt.xlabel("epoch")
  plt.ylabel("loss")
  ensure_dir_exists("figs")
  plt.legend()
  if not tiled:
    figname = "figs/progress-loss-%s" % (data_type)
    plt.savefig(figname)
    plt.close()

if tiled:
  figname = "figs/progress-loss"
  plt.savefig(figname)
  plt.close()
