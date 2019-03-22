"""Script for TCGA cancer type data conversion"""

import pandas
import numpy as np
from common import ensure_dir_exists

clinical_data_file = "data/TCGA_raw/HiSeqV2_PANCAN-2015-02-15/clinical_data"
data_output_dir = "data"

# load clinical data
clinical_data = pandas.read_csv(clinical_data_file, sep='\t', header=0, index_col=0)
clinical_data.index.name = 'sample_id'

# get only cancer types
cancer_type = clinical_data['_primary_disease'].astype('category')

# export to hdf5 format
ensure_dir_exists(data_output_dir)
cancer_type.to_hdf(data_output_dir + "/TCGA_cancertype.h5",
                   'cancer_types', mode='w', format='table')

