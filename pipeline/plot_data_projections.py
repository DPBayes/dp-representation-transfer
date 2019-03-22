import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

from common import ensure_dir_exists
import dataReader


# data types
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
]

# seeds
seeds = [0]


n_projections = (8,2)

#tiled = False
tiled = (n_projections[0], len(data_types)*n_projections[1]) 

#figsize = (12.0, 9.0)
figsize = (6.0, 6.0)

#print(mpl.rcParams['axes.color_cycle'])

if tiled:
  plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

for d, data_type in enumerate(data_types):
  print("data = %s ..." % data_type)
  s = 0
  seed = seeds[s]
  y_train, x_train, y_test, x_test = dataReader.main("%s_%d" % (data_type, seed))
  x_test = x_train
  x = x_test
  x = x[1:1000,:]
  for i in range(n_projections[1] * n_projections[0]):
    if tiled:
      plt.subplot(tiled[1], tiled[0], d * n_projections[1] * tiled[0] + i + 1)
    else:
      plt.figure(figsize=figsize)
    ax = plt.gca()
    a = 2 * i
    b = 2 * i + 1
    plt.plot(x[:,a], x[:,b], '.k', label="data")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel("%d" % a)
    plt.ylabel("%d" % b)

    if not tiled:
      ensure_dir_exists("figs/data_projections")
      figname = "figs/data_projections/%s-%s" % (data_type, alg_id)
      plt.savefig(figname)
      plt.close()

if tiled:
  ensure_dir_exists("figs/data_projections")
  figname = "figs/data_projections/all"
  plt.savefig(figname)
  plt.close()
