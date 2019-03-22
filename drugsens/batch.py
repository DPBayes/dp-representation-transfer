'''Unified framework that support both local and bach jobs
'''

import logging

from common import num_ids_from_args, args_from_id

task_fun = None
num_tasks = None
task_args_ranges = None

def run_task(id):
  """Runs the task of given id (which encodes the generator and the seed)"""
  task_args = args_from_id(id, task_args_ranges)
  logging.info('Running task using args %s', task_args)
  task_fun(task_args)

def run_all_tasks():
  """Runs all tasks"""
  for id in range(num_tasks):
    run_task(id)
  
def run_local():
  """Runs all tasks locally"""
  run_all_tasks()

def run_srun():
  """Runs all tasks using Slurm srun"""
  import sys
  import subprocess
  from batchSettings import independent_srun_args
  script_name = sys.argv[0]
  args = independent_srun_args + ["python", script_name, "all"]
  logging.debug("Calling srun with the following arguments: %s", args)
  subprocess.run(["srun"] + args)

def run_batch(args):
  """Runs all tasks using Slurm array jobs"""
  #import inspect
  import sys
  import subprocess
  import batchSettings
  #script_name = inspect.stack()[0][1]
  script_name = sys.argv[0]
  logging.info("Creating a batch script for sbatch...")
  logging.info("  * an array job with %d tasks", num_tasks)
  logging.debug("  * the task script to run: %s", script_name)
  input = (batchSettings.batchScript
    .replace("{arrayTaskIds}", "0-%d" % (num_tasks-1))
    .replace("{script}", script_name)
  )
  logging.debug("  * batch script content:\n----STARTS----\n%s\n----ENDS----", input)
  logging.info("Calling sbatch with the following arguments: %s", args)
  subprocess.run(["sbatch"] + args, input=input.encode('utf-8'))


def init(task, args_ranges):
  """Sets up the task and argument ranges"""
  global task_fun
  global task_args_ranges
  global num_tasks
  task_fun = task
  task_args_ranges = args_ranges
  num_tasks = num_ids_from_args(task_args_ranges)

def main():
  """Parses arguments and runs locally, batch or specific task"""
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("action", nargs='?', default="local",
                      help="'local' or 'batch' (or task id)",)
  parser.add_argument("--depend",
                      help="array job id to depend on, passed to sbatch",)
  parser.add_argument("--sbatch-depend",
                      help="passed as --depend argument to sbatch",)
  args = parser.parse_args()
  if args.action == "local":
    logging.info('Running algorithms locally...')
    run_local()
  elif args.action == "srun":
    logging.info('Running algorithms sequentially using srun...')
    run_srun()
  elif args.action == "batch":
    logging.info('Running algorithms as batch jobs using Slurm...')
    batch_args = []
    if args.depend:
      args.sbatch_depend = "afterok:" + args.depend
    if args.sbatch_depend:
      batch_args.append("--depend=" + args.sbatch_depend)
    run_batch(batch_args)
  elif args.action == "all":
    logging.info('Running all tasks...')
    run_all_tasks()
  else:
    id = int(args.action)
    logging.info('Running task id %d...', id)
    run_task(id)
