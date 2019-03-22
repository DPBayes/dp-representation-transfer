'''Unified framework that support both local and bach jobs
'''

import logging
import pickle

from common import num_ids_from_args, args_from_id

_task_fun = None

_default_params_file = '/tmp/batch_params.pkl'


# the script that will be feeded to sbatch
# note: the placeholders {extraArgs}, {arrayTaskIds} and {script}
# will be replaced automatically
batchScript = '''#!/bin/bash
{extraArgs}
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --array={arrayTaskIds}

#export CUDA_VISIBLE_DEVICES=""

echo "hostname =" $(hostname)
echo "job id =" $SLURM_ARRAY_JOB_ID
echo "task id =" $SLURM_ARRAY_TASK_ID
echo "script =" {script}

srun python {script} tasks --tasks $SLURM_ARRAY_TASK_ID --params-file {paramsFile}
'''


def _parse_ids(ids):
  import re
  id_list = list()
  ranges = ids.split(',')
  for r in ranges:
    match = re.fullmatch('(\d+)(?:-(\d+)(?::(\d+))?)?', r)
    if match is None:
      raise Exception("invalid task id sequence")
    start, end, step = match.groups()
    start = int(start)
    end = int(end) if end is not None else start
    step = int(step) if step is not None else 1
    id_list.extend(range(start, end + 1, step))
  return id_list

def _run_local(common_params, task_params, task_ids='all', allow_errors=False):
  """Runs tasks locally"""
  if task_ids == 'all':
    ids = range(len(task_params))
  else:
    ids = _parse_ids(task_ids)
  for id in ids:
    logging.info('Running task using args %s', task_params[id])
    try:
      _task_fun(common_params, task_params[id])
    except:
      import sys
      (ex_class, msg, bt) = sys.exc_info()
      logging.error("Task %d failed with the following exception", id)
      logging.error("%s: %s", ex_class.__name__, msg)
      if not allow_errors:
        raise

def _save_params(filename, common_params, task_params):
  with open(filename, 'wb') as f:
    pickle.dump(common_params, f)
    pickle.dump(task_params, f)


def _load_params(filename):
  with open(filename, 'rb') as f:
    common_params = pickle.load(f)
    task_params = pickle.load(f)
  return (common_params, task_params)

def _run_srun(common_params, task_params, params_file, task_ids='all', allow_errors=False,
              slurm_args=None):
  """Runs all tasks using Slurm srun"""
  import sys
  import subprocess
  _save_params(params_file, common_params, task_params)
  script_name = sys.argv[0]
  independent_srun_args = ["--%s=%s" % (k, v) for (k, v) in slurm_args.items()]
  args = independent_srun_args + ["--mpi=pmi2", "python", script_name, "task",
                                  "--tasks", task_ids,
                                  "--params-file", params_file]
  if allow_errors:
    args.append("--allow-errors")
  logging.debug("Calling srun with the following arguments: %s", args)
  subprocess.run(["srun"] + args)

def _run_batch(common_params, task_params, sbatch_args, params_file, task_ids='all',
               slurm_args=None):
  """Runs all tasks using Slurm array jobs"""
  #import inspect
  import sys
  import subprocess
  #import batchSettings2
  _save_params(params_file, common_params, task_params)
  num_tasks = len(task_params)
  #script_name = inspect.stack()[0][1]
  script_name = sys.argv[0]
  logging.info("Creating a batch script for sbatch...")
  if task_ids == 'all':
    logging.info("  * an array job with %d tasks", num_tasks)
    task_ids = "0-%d" % (num_tasks-1)
  else:
    logging.info("  * an array job with tasks %s", task_ids)
    #task_ids = ','.join(map(str, _parse_ids(task_ids)))
  logging.debug("  * the task script to run: %s", script_name)
  batch_srun_args = ["#SBATCH --%s=%s" % (k, v) for (k, v) in slurm_args.items()]
  input = (batchScript
    .replace("{extraArgs}", "\n".join(batch_srun_args))
    .replace("{arrayTaskIds}", task_ids)
    .replace("{script}", script_name)
    .replace("{paramsFile}", params_file)
  )
  logging.debug("  * batch script content:\n----STARTS----\n%s\n----ENDS----", input)
  logging.info("Calling sbatch with the following arguments: %s", sbatch_args)
  subprocess.run(["sbatch"] + sbatch_args, input=input.encode('utf-8'))

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("action", nargs='?', default="local",
                      help="'local' or 'srun' or 'batch' (or task id)")
  parser.add_argument("--wait",
                      help="in the case of batch, wait for all jobs to terminate")
  parser.add_argument("--depend",
                      help="array job id to depend on, passed to sbatch")
  parser.add_argument("--sbatch-depend",
                      help="passed as --depend argument to sbatch")
  parser.add_argument("--params-file",
                      help="task ids to be run")
  parser.add_argument("--tasks", default='all',
                      help="task ids to be run")
  parser.add_argument("--allow-errors", action='store_true',
                      help="continue running next tasks on error (local/srun)")
  args = parser.parse_args()
  return args


def run_tasks(run_args, common_params, task_params, params_file=_default_params_file, slurm_args=None):
  if run_args.action == "local":
    logging.info('Running algorithms locally...')
    _run_local(common_params, task_params, task_ids=run_args.tasks,
               allow_errors=run_args.allow_errors)
  elif run_args.action == "srun":
    logging.info('Running algorithms sequentially using srun...')
    _run_srun(common_params, task_params, params_file, task_ids=run_args.tasks,
              allow_errors=run_args.allow_errors, slurm_args=slurm_args)
  elif run_args.action == "batch":
    logging.info('Running algorithms as batch jobs using Slurm...')
    sbatch_args = []
    if run_args.wait:
      sbatch_args.append("--wait")
    if run_args.depend:
      run_args.sbatch_depend = "afterok:" + run_args.depend
    if run_args.sbatch_depend:
      sbatch_args.append("--depend=" + run_args.sbatch_depend)
    _run_batch(common_params, task_params, sbatch_args, params_file, task_ids=run_args.tasks,
               slurm_args=slurm_args)
  else:
    logging.error("Invalid action '%s'", run_args.action)


def run(task_fun, main_fun=None, common_params=None, task_params=None, params_file=_default_params_file, slurm_args=None):
  """Parses arguments and runs locally, batch or specific task"""
  global _task_fun
  _task_fun = task_fun
  run_args = parse_args()
  if run_args.action == "tasks":
    assert run_args.tasks is not None
    ids = run_args.tasks
    assert run_args.params_file is not None
    params_file = run_args.params_file
    common_params, task_params = _load_params(params_file)
    logging.info('Running task ids %s...', ids)
    _run_local(common_params, task_params, ids)
  else:
    if main_fun is not None:
      assert common_params is None and task_params is None and slurm_args is None
      main_fun(run_args)
    elif params is not None:
      run_tasks(run_args, params, params_file=params_file, slurm_args=slurm_args)
    else:
      assert False, "either main_fun or params must be set"

