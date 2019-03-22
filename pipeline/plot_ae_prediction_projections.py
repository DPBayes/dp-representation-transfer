import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

from common import ensure_dir_exists
import dataReader


# data types
data_types = [
  'twolayers_1-50-50',
  'twolayers_2-50-50',
  'twolayers_3-50-50',
  'twolayers_4-50-50',
  'twolayers_5-50-50',
  'twolayers_7-50-50',
  'twolayers_10-50-50',
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

for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
  #algorithms.append('ae2_' + optimizer)
  algorithms.append('ae2_2x_' + optimizer)
  #algorithms.append('ae2b_8x_' + optimizer)



n_projections = 8

tiled = (n_projections, len(algorithms))

figsize = (6.0, 6.0)

for d, data_type in enumerate(data_types):
  print("data = %s ..." % data_type)
  max_epochs = 0
  algs = []
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))
  for a, alg_id in enumerate(algorithms):
    print("  alg = %s ..." % alg_id)
    s = 0
    seed = seeds[s]
    y_train, x_train, y_test, x_test = dataReader.main("%s_%d" % (data_type, seed))
    x_test = x_train
    pred_filename = 'pred/final-encdec-%s-%d-%s.npy' % (data_type, seed, alg_id)
    x_test_pred = np.load(pred_filename)
    x_test = x_test[1:1000,:]
    x_test_pred = x_test_pred[1:1000,:] 

    for i in range(n_projections):
      #print((a,i))
      plt.subplot(tiled[1], tiled[0], a * tiled[0] + i + 1)
      plt.title("alg = %s" % (alg_id))
      ax = plt.gca()
      u = 2 * i
      v = 2 * i + 1
      plt.plot(np.vstack((x_test[:,u], x_test_pred[:,u])), np.vstack((x_test[:,v], x_test_pred[:,v])), '-', color='lightgrey')
      plt.plot(x_test[:,u], x_test[:,v], '.k', label="data")
      plt.plot(x_test_pred[:,u], x_test_pred[:,v], '.b', label="prediction")
      plt.xlim([-1.2, 1.2])
      plt.ylim([-1.2, 1.2])

  if tiled:
    ensure_dir_exists("figs/predictions")
    figname = "figs/predictions/%s" % data_type
    plt.savefig(figname)
    plt.close()
