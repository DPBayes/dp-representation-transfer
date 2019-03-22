import numpy as np
import matplotlib.pyplot as plt
from common import ensure_dir_exists

alg_pairs = [
  ('pca', 'pca'),
  ('pca', 'ae0_linear'),
  ('ae0_linear', 'ae0_linear'),
  ('ae1_64xR_dropout12_Adam', 'ae1_64xR_dropout12_Adam'),
  ('pca', 'ae1_64xR_dropout12_Adam'),
  ('ae2_32xR_dropout12_Adam', 'ae2_32xR_dropout12_Adam'),
  ('pca', 'ae2_32xR_dropout12_Adam'),
]

ica = False
#ica = True

for a, b in alg_pairs:

  print("%s vs. %s" % (a, b))

  print("  Reading data...")
  x = np.loadtxt("data_repr/repr-10-%s.csv" % a, delimiter=',')
  y = np.loadtxt("data_repr/repr-10-%s.csv" % b, delimiter=',')

  if ica:
    print("  Running FastICA...")
    from sklearn.decomposition import FastICA
    x = FastICA(max_iter=2000).fit_transform(x)
    y = FastICA(max_iter=2000).fit_transform(y)

  print("  Plotting...")

  #plt.figure(figsize=(20,20))
  fig, axes = plt.subplots(nrows=y.shape[1], ncols=x.shape[1], figsize=(20,20),
                           sharex=False, sharey=False)

  for i in range(x.shape[1]):
    for j in range(y.shape[1]):
      #plt.subplot(x.shape[1], y.shape[1], i * y.shape[1] + j + 1)
      #plt.subplot(x.shape[1], y.shape[1], (y.shape[1]-j-1) * x.shape[1] + i + 1,
      #            sharex=True, sharey=True)
      #plt.scatter(x[:,i], y[:,j])
      axes[j, i].scatter(x[:,i], y[:,j])
      if i > 0:
        axes[j, i].tick_params(labelleft='off')
      if j < y.shape[1] - 1:
        axes[j, i].tick_params(labelbottom='off')

  #plt.tight_layout()
  plt.subplots_adjust(wspace=0.1, hspace=0.1,
                      left=0.08, right=0.95, bottom=0.05, top=0.95)

  fig.text(0.5, 0.96, a, ha='center')
  fig.text(0.04, 0.5, b, va='center', rotation='vertical')

  ensure_dir_exists("figs/gdsc_repr_scatter")
  figname = ("figs/gdsc_repr_scatter/gdsc_repr%s_scatter_%s_vs_%s.png" %
            ("_ica" if ica else "", a, b))
  plt.savefig(figname, format='png', dpi=100, bbox_inches='tight')
