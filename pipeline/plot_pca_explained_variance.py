import numpy as np
import matplotlib.pyplot as plt

from common import ensure_dir_exists

# input dataset
data_set = "TCGA_geneexpr_filtered_redistributed"
data_type = 'redistributed_gene_expressions'
#data_set = "TCGA_geneexpr_filtered"
#data_type = 'rnaseq_tpm_rle_log2_gene_expressions'

res_dir = 'res/pca-explained-variance'
res_filename = "%s/%s.txt" % (res_dir, data_set)
expl_var_ratio = np.loadtxt(res_filename)
y = np.cumsum(expl_var_ratio)

cumulative=True
pie=True

print("plotting...")

if cumulative:
  plt.plot(np.arange(1, y.size + 1), y, '-o')
  #plt.yscale('log')
  plt.xscale('log')
  plt.xlabel("no. of components")
  plt.ylabel("explained variance (PCA)")
  plt.ylim(0.0, np.maximum(1.0, y[-1]))
  plt.grid(True)
  plt.title(data_set)
  figname = "figs/pca_explained_variance_cumulative_%s" % data_set
  ensure_dir_exists("figs")
  plt.savefig(figname)
  plt.close()

if pie:
  plt.pie(expl_var_ratio, startangle=90, counterclock=False)
  plt.axis('equal')
  plt.title(data_set)
  figname = "figs/pca_explained_variance_pie_%s" % data_set
  ensure_dir_exists("figs")
  plt.savefig(figname)
  plt.close()
