import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from common import ensure_dir_exists
import dataReader


# data types
data_types = [
  'sphere',
]

# seeds
seeds = [0]

# algorithms
algorithms = [
#  'pca',
#  'ae_linear',
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
#  algorithms.append('ae2_' + optimizer)

for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
  algorithms.append('vae_' + optimizer)


tiled = False
#tiled = (len(algorithms), len(data_types)) 

#figsize = (12.0, 9.0)
figsize = (8.0, 8.0)

if tiled:
  fig = plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for d, data_type in enumerate(data_types):
  print("data = %s ..." % data_type)
  max_epochs = 0
  algs = []
  for a, alg_id in enumerate(algorithms):
    print("  alg = %s ..." % alg_id)
    if tiled:
      ax = fig.add_subplot(tiled[1], tiled[0], d * tiled[0] + a + 1, projection='3d')
    else:
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111, projection='3d')
    plt.title("data = %s, alg = %s" % (data_type, alg_id))
    s = 0
    seed = seeds[s]
    y_train, x_train, y_test, x_test = dataReader.main("%s_%d" % (data_type, seed))
    x_test = x_train
    pred_filename = 'pred/final-encdec-%s-%d-%s.npy' % (data_type, seed, alg_id)
    x_test_pred = np.load(pred_filename)
    x_test = x_test[1:100,:]
    x_test_pred = x_test_pred[1:100,:]

    for xpair in zip(x_test, x_test_pred):
      xs, ys, zs = zip(*xpair)
      ax.plot(xs, ys, zs, '-', color='lightgrey')
    #ax.plot(xs=np.vstack((x_test[:,0], x_test_pred[:,0])),
    #        ys=np.vstack((x_test[:,1], x_test_pred[:,1])),
    #        zs=np.vstack((x_test[:,2], x_test_pred[:,2])),
    #        linestyle='-', color='lightgrey')
    ax.plot(x_test[:,0], x_test[:,1], x_test[:,2], '.k', label="data")
    ax.plot(x_test_pred[:,0], x_test_pred[:,1], x_test_pred[:,2], '.b', label="prediction")

    ax.legend()

    if not tiled:
      #ensure_dir_exists("figs/predictions")
      #figname = "figs/predictions/%s-%s" % (data_type, alg_id)
      #plt.savefig(figname)
      #plt.close()
      fig.show()

if tiled:
  #ensure_dir_exists("figs/predictions")
  #figname = "figs/predictions/all"
  #plt.savefig(figname)
  #plt.close()
  fig.show()

plt.show()