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

priv_cancertypes = [
  'breast invasive carcinoma',   # 1212 / 10534
  'kidney clear cell carcinoma', # 603 / 10534
  ]

data_in_dir = "data"
data_out_dir = "data"

# *** Configuration ends ***

logging.basicConfig(level=logging.INFO)


logging.info("Loading TCGA...")
gene_expr = pandas.read_hdf(data_in_dir + "/TCGA_geneexpr.h5",
                            'rnaseq_tpm_rle_log2_gene_expressions')
logging.info(" * loaded, size: %d samples, %d genes" % gene_expr.shape)

# filter out genes whose median in TCGA less than 0.0
logging.info("Filtering out genes with low expressions in TCGA...")
low_expr = (np.median(gene_expr, axis=0) < 0.0)
gene_expr = gene_expr.iloc[:,~low_expr]
logging.info(" * %d of %d remaining (%d removed)" %
             (sum(~low_expr), low_expr.size, sum(low_expr)))


logging.info("Loading cancer types...")
cancer_type = pandas.read_hdf(data_in_dir + "/TCGA_cancertype.h5",
                              'cancer_types')
assert np.array_equal(gene_expr.index, cancer_type.index)

# split
logging.info("Splitting...")
priv = cancer_type.isin(priv_cancertypes)
logging.info(" * %d private samples, %d public samples (of %d total)" %
             (sum(priv), sum(~priv), priv.size))


# save
logging.info("Saving...")
gene_expr[~priv].to_hdf(data_out_dir + "/TCGA_split_pub_geneexpr.h5",
    'rnaseq_tpm_rle_log2_gene_expressions', mode='w')
cancer_type[~priv].to_hdf(data_out_dir + "/TCGA_split_pub_cancertype.h5",
    'cancer_types', mode='w', format='table')
gene_expr[priv].to_hdf(data_out_dir + "/TCGA_split_priv_geneexpr.h5",
    'rnaseq_tpm_rle_log2_gene_expressions', mode='w')
cancer_type[priv].to_hdf(data_out_dir + "/TCGA_split_priv_cancertype.h5",
    'cancer_types', mode='w', format='table')

