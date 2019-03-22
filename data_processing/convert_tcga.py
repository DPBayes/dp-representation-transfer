"""Script for TCGA data conversion"""

import pandas
import numpy as np
from common import ensure_dir_exists
import readline # to get rpy2 working under conda env
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
edger = rpy2.robjects.packages.importr('edgeR')
import logging

logging.basicConfig(level=logging.INFO)

geneexpr_file = "data/TCGA_raw/tcga_RSEM_gene_tpm"
gene_info_file = "data/TCGA_raw/gencode.v23.annotation.gene.probeMap"
clinical_data_file = "data/TCGA_raw/PANCAN_clinicalMatrix"
data_output_dir = "data"


# load gene expression data
logging.info("Loading gene expression data...")
geneexpr = pandas.read_csv(geneexpr_file, sep='\t', header=0, index_col=0)
logging.info(" * loaded, size: %d genes, %d samples" % geneexpr.shape)

logging.info("Loading and applying gene names...")
# load gencode gene ids to gene names conversion table
gene_info = pandas.read_csv(gene_info_file, sep='\t', header=0, index_col=0)

# find index genes by their names instead of ensembl ids,
# Note: some ensembl ids have the same gene id! Example:
#   id                      gene chrom  chromStart  chromEnd strand
#   ENSG00000227372.10  TP73-AS1  chr1     3735601   3747336      -
#   ENSG00000276189.1   TP73-AS1  chr1     3736943   3737103      +
geneexpr.index = pandas.Index(gene_info.loc[geneexpr.index, "gene"].values, name='gene_name')

# invert log transformation
logging.info("Inverting log_2(x+0.001) transformation...")
tpm = np.maximum(2 ** geneexpr - 0.001, 0)
del geneexpr

# sum the values with the same gene name
logging.info("Summing the values of genes with the same gene id...")
tpm = tpm.groupby(tpm.index, sort=False).sum()
logging.info(" * done, %d genes left" % tpm.shape[0])

# filter out low expression genes
logging.info("Filtering out low expression genes...")
not_low = (np.sum(tpm > 1, axis=1) >= 2)
tpm = tpm[not_low]
logging.info(" * filtered, %d genes left" % tpm.shape[0])

# filter out samples with zero expression
# tpm = tpm.loc[:, np.sum(tpm, axis=0) != 0]
assert not np.any(np.sum(tpm, axis=0) == 0)

# calculate normalization factors
logging.info("Calculating normalization factors...")
#r_data = edger.DGEList(tpm.values, genes=tpm.index)
#r_data = edger.calcNormFactors(r_data, method="RLE")
#norm_factors = np.array(r_data.rx2('samples').rx2('norm.factors'))
norm_factors = np.array(edger.calcNormFactors(tpm.values, method="RLE"))

# apply normalization factors
logging.info("Applying normalization factors...")
geneexpr = tpm * norm_factors
del tpm

# redo log-transformation
logging.info("Redoing log transformation...")
geneexpr = np.log2(geneexpr + 0.001)

# transpose so that columns=genes and rows=samples
geneexpr = geneexpr.transpose()
geneexpr.index.name = 'sample_id'

# load clinical data
logging.info("Loading clinical data...")
clinical_data = pandas.read_csv(clinical_data_file, sep='\t', header=0, index_col=0)
clinical_data.index.name = 'sample_id'
logging.info(" * loaded %d samples" % clinical_data.index.size)

# get only cancer types
cancer_type = clinical_data['_primary_disease'].astype('category')

# find and select commmon sample ids
logging.info("Taking only common samples in the datasets...")
common_samples = cancer_type.index.intersection(geneexpr.index)
cancer_type = cancer_type.loc[common_samples]
geneexpr = geneexpr.loc[common_samples]
logging.info(" * %d common samples" % common_samples.size)

# export to hdf5 format
logging.info("Saving to HDF5...")
ensure_dir_exists(data_output_dir)
geneexpr.to_hdf(data_output_dir + "/TCGA_geneexpr.h5",
                'rnaseq_tpm_rle_log2_gene_expressions', mode='w')
cancer_type.to_hdf(data_output_dir + "/TCGA_cancertype.h5",
                   'cancer_types', mode='w', format='table')
