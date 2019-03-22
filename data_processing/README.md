# Data pre-processing tools for TCGA and GDSC datasets

## Script order

Note: check the datafolder settings in each script before running

1. `convert_tcga.py`: raw TCGA datasets (gene expressions + cancer types)
2. `convert_gdsc.py`: raw GDSC datasets (gene expressions + drug responses)
3. `unify_tcga_gdsc_datas.py`: filters genes and unifies the marginal distribution(s) of TCGA and GDSC datasets
4. `split_tcga_cancertypes.py`: create separate datasets for different cancer types


## TCGA dataset

Downloaded from <https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)>

* gene expressions (tpm transformed) + id/gene mappings:
  "TOIL Gene RSEM tpm"
  Page: <https://xenabrowser.net/datapages/?dataset=tcga_RSEM_gene_tpm&host=https://toil.xenahubs.net>
  Note: need also the annotations
  Version: 2016-09-01

* clinical data (includes cancer types):
  "Phenotypes"
  Page: 
<https://xenabrowser.net/datapages/?dataset=TCGA_phenotype_denseDataOnlyDownload.tsv&host=https%3A%2F%2Fpancanatlas.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443>
    Version: 2016-11-18


## GDSC dataset

Datasets downloaded from <http://www.cancerrxgene.org/downloads>

* drug responses:
  "Drug|Preprocessed|Cell lines/Drugs|log(IC50) and AUC values"
  <ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/v17_fitted_dose_response.xlsx>

* gene expressions:
  "Expression|Preprocessed|Cell lines|RMA normalised expression data for Cell lines"
  <https://dl.dropboxusercontent.com/u/11929126/sanger1018_brainarray_ensemblgene_rma.txt.gz>

Extracted `sanger1018_brainarray_ensemblgene_rma.txt`.
Converted `v17_fitted_dose_response.xlsx` to `v17_fitted_dose_response.csv`

Conversion table between ensembl gene ids and gene names downloaded from
<http://www.ensembl.org/biomart/martview/> as follows
1. Select: Ensembl Genes 88 -> Human genes (GRCh38.p10)
2. Edit Attributes: Select only "Gene stable ID" and "Gene name".
3. Click results.
4. Export to csv. Rename to `ensembl_gene_ids.csv`

