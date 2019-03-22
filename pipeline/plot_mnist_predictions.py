import os.path
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as plt.transforms 

from common import ensure_dir_exists
import dataReader


# data types
data_type = 'mnist'

repr_dim = 16

# seeds
seeds = [0]

# samples
samples = range(15)

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
#  'vae_torch',
  'vae_gs',
  'vae_gi',
  'vae_ps',
  'vae_pi',
  'vae_torch_gs',
  'vae_torch_gi',
  'vae_torch_ps',
  'vae_torch_pi',
#  'vae_pi_e4_d16',
#  'vae_pi_e4e1_d1d8',
#  'vae_pi_e4e1_d1d8_prelu',
#  'vae_pi_e4e1_d1d8_do2100',
#  'vae_pi_e4e1_d1d8_do52',
]

#for optimizer in ['Adam']:
#  algorithms.append('ae1_' + optimizer)

#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
#  #algorithms.append('ae2_2x_' + optimizer)
#  algorithms.append('ae2b_8x_' + optimizer)

#for optimizer in ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']:
  #algorithms.append('ae2_2x_' + optimizer)
  #algorithms.append('vae_' + optimizer)



#tiled = False
tiled = (1 + 2*len(algorithms), len(samples))

#figsize = (12.0, 9.0)
figsize = (2.0, 2.0)

#print(mpl.rcParams['axes.color_cycle'])

plt.figure(figsize=(tiled[0]*figsize[0], tiled[1]*figsize[1]))

print("data = %s ..." % data_type)
max_epochs = 0
algs = []

print("  original ...")
data_filename = 'data/generated/%s.npy' % (data_type)
x_test = np.load(data_filename)
for s, sample in enumerate(samples):
  plt.subplot(tiled[1], tiled[0], s * tiled[0] + 1)
  plt.axis('off')
  #plt.title("alg = %s" % (data_type, alg_id))
  #ax = plt.gca()
  plt.imshow(x_test[sample,:].reshape((28,28)), cmap='gray')
  if s == 0:
    plt.title("original")

s = 0
seed = seeds[s]

for a, alg_id in enumerate(algorithms):
  print("  alg = %s ..." % alg_id)
  #pred_filename = 'pred/final-encdec-%s-%d-%s.npy' % (data_type, seed, alg_id)
  #pred_rand_filename = 'pred/final-encdec-rand-%s-%d-%s.npy' % (data_type, seed, alg_id)
  pred_filename = 'pred/final-encdec-%s-r%d-s%d-%s.npy' % (data_type, repr_dim, seed, alg_id)
  pred_rand_filename = 'pred/final-encdec-rand-%s-r%d-s%d-%s.npy' % (data_type, repr_dim, seed, alg_id)
  x_test_pred = np.load(pred_filename)
  x_test_pred_rand = np.load(pred_rand_filename)
  for s, sample in enumerate(samples):
    plt.subplot(tiled[1], tiled[0], s * tiled[0] + 2*a + 2)
    plt.axis('off')
    plt.imshow(x_test_pred[sample,:].clip(0,1).reshape((28,28)), cmap='gray')
    if s == 0:
      plt.title(alg_id)
    plt.subplot(tiled[1], tiled[0], s * tiled[0] + 2*a + 3)
    plt.axis('off')
    plt.imshow(x_test_pred_rand[sample,:].clip(0,1).reshape((28,28)), cmap='gray')

ensure_dir_exists("figs/predictions")
#figname = "figs/predictions/%s" % (data_type)
figname = "figs/predictions/%s-r%d" % (data_type, repr_dim)
plt.savefig(figname)
plt.close()
