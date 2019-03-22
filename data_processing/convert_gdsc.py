"""Script for GDSC data conversion"""

import pandas
import numpy as np
from common import ensure_dir_exists
import logging

logging.basicConfig(level=logging.INFO)

geneexpr_file = "data/GDSC/sanger1018_brainarray_ensemblgene_rma.txt"
ensembl_name_file = "data/GDSC/ensembl_gene_ids.csv"
drugres_file = "data/GDSC/v17_fitted_dose_response.csv"
data_output_dir = "data"

# load gene expression data
# note: Some (4) samples appear twice, for instance 1503362.
#       The second appearance is renamed to 1503362.1, which is then later ignored
#       when finding common samples with the gene expression data.
logging.info("Loading gene expression data...")
geneexpr = pandas.read_csv(geneexpr_file, sep='\t', header=0, index_col=0)
logging.info(" * loaded, size: %d genes, %d samples" % geneexpr.shape)

# load ensembl gene ids to gene names conversion table
logging.info("Loading and applying gene names...")
gene_names = pandas.read_csv(ensembl_name_file, header=0, index_col=0)

# find index genes by their names instead of ensembl ids,
# drop those that do not have a name
common_gene_ids = geneexpr.index.intersection(gene_names.index)
geneexpr = geneexpr.loc[common_gene_ids]
geneexpr.index = pandas.Index(gene_names.loc[common_gene_ids, "Gene name"].values, name='gene_name')
logging.info(" * names for %d genes found" % geneexpr.shape[0])

# transpose so that columns=genes and rows=samples
geneexpr = geneexpr.transpose()
geneexpr.index.name = 'sample_id'

# load drug response data
logging.info("Loading drug response data...")
drugres_data = pandas.read_csv(drugres_file)
drugres = drugres_data.pivot(index='COSMIC_ID', columns='DRUG_ID',
                             values='LN_IC50')
drugres.index = pandas.Index(drugres.index.map(str), name='sample_id')
drugres.columns.name = 'drug_id'
logging.info(" * loaded %d samples of %d drugs" % drugres.shape)

# find and select commmon sample ids
logging.info("Taking only common samples in the datasets...")
common_samples = drugres.index.intersection(geneexpr.index)
drugres = drugres.loc[common_samples]
geneexpr = geneexpr.loc[common_samples]
logging.info(" * %d common samples" % common_samples.size)

# export to hdf5 format
logging.info("Saving to HDF5...")
ensure_dir_exists(data_output_dir)
geneexpr.to_hdf(data_output_dir + "/GDSC_geneexpr.h5",
                'rma_gene_expressions', mode='w')
drugres.to_hdf(data_output_dir + "/GDSC_drugres.h5",
               'drug_responses', mode='w')


#drugres_data['DRUG_ID'].unique()
#drugres_data['COSMIC_ID'].unique()
#geneexpr_data.index.values

#geneexpr = pandas.read_hdf("GDSC_geneexpr.h5", 'rma_gene_expressions')

