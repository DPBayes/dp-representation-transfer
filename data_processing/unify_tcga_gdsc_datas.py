"""
Script for unigying TCGA and GDSC gene expression datasets.
 * filters out genes with low expression leves in TCGA
 * takes only genes that appear in both datasets
 * redistributes one according to the other
"""

import numpy as np
import pandas
import logging

# *** Configuration starts ***

redistribute_tcga = False
redistribute_gdsc = True

redistribute_each_column = False
redistribute_full_data = True

# *** Configuration ends ***

logging.basicConfig(level=logging.INFO)

assert redistribute_tcga != redistribute_gdsc
assert redistribute_each_column != redistribute_full_data

logging.info("Loading TCGA...")
tcga = pandas.read_hdf("data/TCGA_geneexpr.h5",
                       'rnaseq_tpm_rle_log2_gene_expressions')
#                       'rnaseq_log2_gene_expressions')
logging.info(" * loaded, size: %d samples, %d genes" % tcga.shape)

logging.info("Loading GDSC...")
gdsc = pandas.read_hdf("data/GDSC_geneexpr.h5",
                       'rma_gene_expressions')
logging.info(" * loaded, size: %d samples, %d genes" % gdsc.shape)

# filter out genes whose median in TCGA less than 0.0
logging.info("Filtering out genes with low expressions in TCGA...")
low_expr = (np.median(tcga, axis=0) < 0.0)
tcga = tcga.iloc[:,~low_expr]
logging.info(" * %d of %d remaining (%d removed)" %
             (sum(~low_expr), low_expr.size, sum(low_expr)))

# get only common genes
logging.info("Take only genes appearing in both TCGA and GDSC...")
common_genes = tcga.columns.intersection(gdsc.columns)
tcga = tcga.loc[:,common_genes]
gdsc = gdsc.loc[:,common_genes]
logging.info(" * %d genes remaining" % (common_genes.size))

def redistribute_all(x, ref):
  """Redistribute x according to the distributions of ref"""
  ref_sorted = np.sort(ref, axis=None)
  x_order = np.argsort(x, axis=None)
  x_ranks = np.empty(x_order.shape)
  x_ranks[x_order] = np.linspace(0, ref_sorted.size - 1, x_order.size)
  a = x_ranks - np.floor(x_ranks)
  x_redistributed = ((1-a) * ref_sorted[np.floor(x_ranks).astype(int)] + 
                        a * ref_sorted[np.ceil(x_ranks).astype(int)])
  return x_redistributed.reshape(x.shape)

def redistribute_columns(x, ref):
  """Redistribute the columns of x according to the distributions of the columns of ref"""
  assert x.shape[1] == ref.shape[1]
  for g in range(x.shape[1]):
    x[:,g] = redistribute_all(x[:,g], ref[:,g])
  return x

if redistribute_gdsc and not redistribute_tcga:
  # make gdsc data to follow tcga distribution
  logging.info("Redistributing GDSC according to TCGA...")
  if redistribute_each_column and not redistribute_full_data:
    gdsc_mat = redistribute_columns(gdsc.as_matrix(), tcga.as_matrix())
  elif redistribute_full_data and not redistribute_each_column:
    gdsc_mat = redistribute_all(gdsc.as_matrix(), tcga.as_matrix())
  else:
    assert False
  gdsc = pandas.DataFrame(gdsc_mat, index=gdsc.index, columns=gdsc.columns)
  # save
  logging.info("Saving...")
  tcga.to_hdf("data/TCGA_geneexpr_filtered.h5",
              'rnaseq_tpm_rle_log2_gene_expressions', mode='w')
  gdsc.to_hdf("data/GDSC_geneexpr_filtered_redistributed.h5",
              'redistributed_gene_expressions', mode='w')
elif redistribute_tcga and not redistribute_gdsc:
  # make tcga data to follow gdsc distribution
  logging.info("Redistributing TCGA according to GDSC...")
  if redistribute_each_column and not redistribute_full_data:
    tcga_mat = redistribute_columns(tcga.as_matrix(), gdsc.as_matrix())
  elif redistribute_full_data and not redistribute_each_column:
    tcga_mat = redistribute_all(tcga.as_matrix(), gdsc.as_matrix())
  else:
    assert False
  tcga = pandas.DataFrame(tcga_mat, index=tcga.index, columns=tcga.columns)
  # save
  logging.info("Saving...")
  tcga.to_hdf("data/TCGA_geneexpr_filtered_redistributed.h5",
              'redistributed_gene_expressions', mode='w')
  gdsc.to_hdf("data/GDSC_geneexpr_filtered.h5",
              'rma_gene_expressions', mode='w')
else:
  assert False


