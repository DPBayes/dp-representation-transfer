import numpy as np
import logging
import batch
from common import ensure_dir_exists

# logging configuration
logging.basicConfig(level=logging.INFO)

# input datasets
data_set = "TCGA_geneexpr_filtered_redistributed"
repr_dims = [10]

normalize_data = True
#validation_split = 0
validation_split = 0.2

algorithms = []

# PCA
algorithms.append((
    'pca',
    lambda repr_dim, alg_id='pca':
      __import__('models.pca').pca.PCA().load("repr_models/%s-%d-%s" %
                                          (data_set, repr_dim, alg_id))
  ))

# autoencoders
optimizer = 'Adam'
for alg_id in [
                'ae0_linear',
                'ae1_64xR_dropout12_%s' % (optimizer),
                'ae2_32xR_dropout12_%s' % (optimizer),
                'ae3_32xR_dropout12_%s' % (optimizer),
              ]:
  algorithms.append((
      alg_id,
      lambda repr_dim, alg_id=alg_id:
        __import__('models.ae').ae.AE().load("repr_models/%s-%d-%s" %
                                            (data_set, repr_dim, alg_id))
    ))

ica = False
#ica = True

assert ica == False # not implemented


# the task function that is run with each argument combination
def task(args):
  repr_dim, (alg_id, load_model) = args
  logging.info("representation size = %d, algorithm = %s", repr_dim, alg_id)

  # read the GDSC gene expression data
  logging.info("Reading gene expression data...")
  import pandas
  data = pandas.read_hdf("data/%s.h5" % (data_set),
                      'redistributed_gene_expressions')
  x = data.as_matrix()
  logging.info(" * data shape: %d x %d" % x.shape)

  # normalize the input to _total_ unit variance and zero mean
  if normalize_data: 
    x -= np.mean(x)
    x /= np.std(x)
  
  # init rng  
  np.random.seed(0)

  # load the model
  logging.info("Loading the model...")
  alg = load_model(repr_dim)
  
  # get the representation
  logging.info("Computing the representation or size %d..." % (repr_dim))
  x_repr = alg.encode(x)

  # variance of each representation component
  #repr_vars = np.var(x_repr, axis=0)

  repr_avg = np.mean(x_repr, axis=0)

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(nrows=repr_dim, ncols=1, figsize=(16,10),
                           sharex=True, sharey=True)

  logging.info("Computing and plotting projections...")
  x_repr_onedim = np.empty(x_repr.shape)
  for i in range(repr_dim):
    logging.info("  * component %d/%d" % (i+1, repr_dim))
    x_repr_onedim[:,:] = repr_avg
    x_repr_onedim[:,i] = x_repr[:,i]
    repr_proj = alg.decode(x_repr_onedim)
    proj_std = np.std(repr_proj, axis=0)
    #plt.subplot(repr_dim, 1, i+1)
    axes[i].bar(np.arange(x.shape[1]), proj_std, color='b', edgecolor='none')
    #axes[i].bar(np.arange(50), proj_std[0:50], color='b', edgecolor='none')
    plt.ylabel("projection std")
    plt.xlabel("gene")
    #plt.title("repr component %d" % i)

  ensure_dir_exists("figs/tcga_repr_projections")
  figname = "figs/tcga_repr_projections/d%d_%s.png" % (repr_dim, alg_id)
  plt.savefig(figname, format='png', dpi=200)
  plt.close(fig)

########## MAIN ##########

# init and run
batch.init(task=task, args_ranges=(repr_dims, algorithms))
batch.main()
