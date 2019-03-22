

# the script that will be feeded to sbatch
# note: the placeholders {arrayTaskIds} and {script} will be replaced automatically
batchScript = '''#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=16G
##SBATCH --mem=4G
##SBATCH --partition=batch
#SBATCH --partition=short
##SBATCH --partition=gpushort
##SBATCH --gres=gpu:1
#SBATCH --constraint=hsw
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --array={arrayTaskIds}

#export CUDA_VISIBLE_DEVICES=""

THEANO_BASE_COMPILEDIR=$TMPDIR/$USER/theano/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
export THEANO_FLAGS=device=gpu,floatX=float32,base_compiledir=$THEANO_BASE_COMPILEDIR

echo "hostname =" $(hostname)
echo "job id =" $SLURM_ARRAY_JOB_ID
echo "task id =" $SLURM_ARRAY_TASK_ID
echo "script =" {script}

#export KERAS_BACKEND=tensorflow
srun python {script} task --task $SLURM_ARRAY_TASK_ID

#export KERAS_BACKEND=theano
#srun --mpi=pmi2 python {script} $SLURM_ARRAY_TASK_ID

'''

# these will be given to srun when it is called independently (not as a part of sbatch)
independent_srun_args = [
    "--time=1:00:00",
    "--mem=16G",
    "--partition=short",
    #"--partition=gpushort",
    #"--gres=gpu:1",
    #"--gres=gpu:teslak80:1",
    "--constraint=hsw",
    "--mpi=pmi2",
  ]


## parameters to srun
## note: "--array" parameter will be added automatically
#sbatchParams = '''
##SBATCH --time=04:00:00
##SBATCH --mem=8G
##SBATCH --partition=gpushort
##SBATCH --gres=gpu:teslak80:1
##SBATCH -o generate.out
##SBATCH -e generate.err
#'''
#
## will be added to the generated sbatch file before "srun" 
#sbatchBeforeSrun = '''
##export CUDA_VISIBLE_DEVICES=""
#THEANO_BASE_COMPILEDIR=$TMPDIR/$USER/theano/keras-test
#export THEANO_FLAGS=device=gpu,floatX=float32,base_compiledir=$THEANO_BASE_COMPILEDIR
#
#echo "hostname =" $(hostname)
#
#source $PYENVDIR/bin/activate
#'''
#
#sbatchSrun = '''
#export KERAS_BACKEND=tensorflow
#srun python {script} $SLURM_ARRAY_TASK_ID
##export KERAS_BACKEND=theano
##srun --mpi=pmi2 python {script} $SLURM_ARRAY_TASK_ID
#'''
#
## will be added to the generated sbatch file after "srun" 
#sbatchAfterSrun = '''
#deactivate
#'''




