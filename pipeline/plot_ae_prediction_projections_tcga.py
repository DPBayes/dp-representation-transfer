import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

from common import ensure_dir_exists
import dataReader


# data types
data_sets = [
  'geneMatrix',
]

#input_dim = None
#input_dim = 2000
input_dim = 50

# the dimension of the representation to be learned
#repr_dims = [10, 50, 100, 200, 500, 1000]
repr_dims = [5, 10, 20, 30]

# algorithms
algorithms = [
  'pca',
  'ae0_linear',
]

for optimizer in ['Adagrad', 'Adam', 'Adamax']:
  algorithms.append('ae1_2x_' + optimizer)


n_projections = 8

tiled = (n_projections, len(algorithms))

figsize = (6.0, 6.0)

for d, data_set in enumerate(data_sets):
  print("data = %s , input_dim = %s ..." % (data_set, input_dim))
  x = getHDF5data("data/%s.h5" % (data_set), True, True)[0]
  # transpose and redo the log transform
  x = x.T
  x = np.log1p(x)
  # make data size a multiple of batch size (drop extra rows)
  n = (x.shape[0] // batch_size) * batch_size
  x = x[0:n, :]
  # optionally reduce dimensionality
  if input_dim is not None:
    x = x[:, 0:input_dim]
  
  for r, repr_dim in enumerate(repr_dims):
    print(" repr_dim = %s ..." % repr_dim)
    max_epochs = 0
    algs = []
    plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))
    for a, alg_id in enumerate(algorithms):
      print("  alg = %s ..." % alg_id)
      pred_filename = 'pred/final-encdec-%s-%s-%d-%s.npy' % (data_set, input_dim, repr_dim, alg_id)
      x_pred = np.load(pred_filename)
      x = x[1:1000,:]
      x_pred = x_pred[1:1000,:] 

      for i in range(n_projections):
        #print((a,i))
        plt.subplot(tiled[1], tiled[0], a * tiled[0] + i + 1)
        plt.title("alg = %s" % (alg_id))
        ax = plt.gca()
        u = 2 * i
        v = 2 * i + 1
        plt.plot(np.vstack((x[:,u], x_pred[:,u])), np.vstack((x[:,v], x_pred[:,v])), '-', color='lightgrey')
        plt.plot(x[:,u], x[:,v], '.k', label="data")
        plt.plot(x_pred[:,u], x_pred[:,v], '.b', label="prediction")
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])

    if tiled:
      ensure_dir_exists("figs/predictions")
      figname = "figs/predictions/%s-%s-%d" % (data_set, input_dim, repr_dim)
      plt.savefig(figname)
      plt.close()
