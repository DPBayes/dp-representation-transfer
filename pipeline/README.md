
# Codes for combining deep learning based representation learning with DP

## Content of this directory

* scripts for testing plain representation learning:
  * `test_repr_*`
  * `test_pca_*`
* representation learning for DP drug sensitivity prediction:
  * `learn_tcga_reduced_repr_model.py`
  * `learn_tcga_reduced_repr_aux_model.py`
  * `make_gdsc-repr.py`
* result plotting scripts:
  * `plot_*`
* modules used in scripts:
  * `batch`: for launching multiple tests/jobs with different parameter combinations
  * `batchSettings`: slurm settings
  * `common`: miscellaneous functions etc.
  * `dataReader`
  * `readHDF5`
* models for representation learning in `models` subdirectory
  * `models.ae`: Autoencoder.
  * `models.pca`: PCA.
  * `models.nncommon`: Common functions and other stuff for deep learning (Keras).
* backend function implementations for Keras in `backend` subdirectory
* old stuff in `old_stuff`

## Directories for data / results / figures

Scripts can be run from any directory. The following subdirectories of the current working directory will be used:
* `data`: Should have the required data files. Generated data will also be placed here.
* `figs`: Plotting scripts will output figures here.
* `log`: Logs of testing scripts.
* `pred`: Predictions made by models.
* `repr_models`: Learned representation models. 
* `res`: Results from testing scripts.

## Running scripts that use `batch` module

The scripts that use `batch` module can launch multiple jobs with different parameter combinations. be run either locally
   * To run sequentially locally:
     `python (path/to/)scriptname.py local`
   * To run sequentially on a single server using slurm:
     1. Edit `batchSettings.py` to configure slurm settings
     2. Launch by running:
        `python (path/to/)scriptname.py srun`
   * To run as parallel batch jobs using slurm:
     1. Edit `batchSettings.py` to configure slurm settings
     2. Send batch jobs by running:
        `python (path/to/)scriptname.py batch`


## Plain representation learning with generated data

0. (Generate the datasets using the scripts in `data_generation` subdirectory.)
1. Edit `test_repr_learning*.py` to select the datasets and configure algorithms.
2. Run `test_repr_learning*.py` either locally or as batch job (see above instructions for using `batch` module)
3. Generate figures from the results. These scripts must be first edited to use correct result files.
   * To plot the progresse of the MSE, edit and run `plot_mse_progress.py`.
   * To plot the progresse of weights, edit and run `plot_weight_progress.py`.
   * etc.

* The algorithms are located in `models` subdirectory

## Plain representation learning with TCGA data

This goes as above but obviously data generation is not needed and you should only use scripts (`test_*` and `plot_*`) that have word `tcga` in their name.

## Representation learning for DP drug sensitivity prediction

0. Preprocess the datasets. See the `../data_processing` directory.
1. Learn representation mappings (models) by configuring and running (using `batch`) `learn_tcga_reduced_repr_model.py` (optimizes only reconstruction error) or `learn_tcga_reduced_repr_aux_model.py` (optimizes both reconstruction and auxiliary (cancer type) prediction errors).
2. Apply mappings to GDSC by configuring and running `make_gdsc-repr.py`.
3. Run the drug sensitivity prediction tests using the learned representation. See the `drugsens` directory.


