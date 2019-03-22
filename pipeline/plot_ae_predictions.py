import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

from common import ensure_dir_exists
import dataReader


# data types
data_types = [
  'parabola',
  'u',
  'circle',
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

#for optimizer in ['Adam']:
#  algorithms.append('ae1_' + optimizer)

#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  #algorithms.append('ae2_2x_' + optimizer)
#  algorithms.append('ae2b_8x_' + optimizer)

for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
  #algorithms.append('ae2_2x_' + optimizer)
  algorithms.append('vae_' + optimizer)



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
    seed = seeds[s]
    y_train, x_train, y_test, x_test = dataReader.main("%s_%d" % (data_type, seed))
    x_test = x_train
    pred_filename = 'pred/final-encdec-%s-%d-%s.npy' % (data_type, seed, alg_id)
    x_test_pred = np.load(pred_filename)
    x_test = x_test[1:1000,:]
    x_test_pred = x_test_pred[1:1000,:] 

    ax = plt.gca()
    plt.plot(np.vstack((x_test[:,0], x_test_pred[:,0])), np.vstack((x_test[:,1], x_test_pred[:,1])), '-', color='lightgrey')
    plt.plot(x_test[:,0], x_test[:,1], '.k', label="data")
    plt.plot(x_test_pred[:,0], x_test_pred[:,1], '.b', label="prediction")

    plt.legend()

    if not tiled:
      ensure_dir_exists("figs/predictions")
      figname = "figs/predictions/%s-%s" % (data_type, alg_id)
      plt.savefig(figname)
      plt.close()

if tiled:
  ensure_dir_exists("figs/predictions")
  figname = "figs/predictions/all"
  plt.savefig(figname)
  plt.close()
