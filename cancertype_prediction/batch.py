'''Unified framework that support both local and bach jobs
'''

import logging

from common import num_ids_from_args, args_from_id

_task_fun = None
_num_tasks = None
_task_args_ranges = None

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

def _run_task(id):
  """Runs the task of given id (which encodes the generator and the seed)"""
  task_args = args_from_id(id, _task_args_ranges)
  logging.info('Running task using args %s', task_args)
  _task_fun(task_args)
  
def _run_local(task_ids='all', allow_errors=False):
  """Runs tasks locally"""
  #run_all_tasks()
  if task_ids == 'all':
    ids = range(_num_tasks)
  else:
    ids = _parse_ids(task_ids)
  for id in ids:
    if allow_errors:
      try:
        _run_task(id)
      except:
        import sys
        (ex_class, msg, bt) = sys.exc_info()
        logging.error("Task %d failed with the following exception", id)
        logging.error("%s: %s", ex_class.__name__, msg)
    else:
      _run_task(id)

def _run_srun(task_ids='all', allow_errors=False):
  """Runs all tasks using Slurm srun"""
  import sys
  import subprocess
  from batchSettings import independent_srun_args
  script_name = sys.argv[0]
  args = independent_srun_args + ["python", script_name, "local", "--tasks", task_ids]
  if allow_errors:
    args.append("--allow-errors")
  logging.debug("Calling srun with the following arguments: %s", args)
  subprocess.run(["srun"] + args)

def _run_batch(args, task_ids='all'):
  """Runs all tasks using Slurm array jobs"""
  #import inspect
  import sys
  import subprocess
  import batchSettings
  #script_name = inspect.stack()[0][1]
  script_name = sys.argv[0]
  logging.info("Creating a batch script for sbatch...")
  if task_ids == 'all':
    logging.info("  * an array job with %d tasks", _num_tasks)
    task_ids = "0-%d" % (_num_tasks-1)
  else:
    logging.info("  * an array job with tasks %s", task_ids)
    #task_ids = ','.join(map(str, _parse_ids(task_ids)))
  logging.debug("  * the task script to run: %s", script_name)
  input = (batchSettings.batchScript
    .replace("{arrayTaskIds}", task_ids)
    .replace("{script}", script_name)
  )
  logging.debug("  * batch script content:\n----STARTS----\n%s\n----ENDS----", input)
  logging.info("Calling sbatch with the following arguments: %s", args)
  subprocess.run(["sbatch"] + args, input=input.encode('utf-8'))

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
  parser.add_argument("--tasks", default='all',
                      help="task ids to be run")
  parser.add_argument("--task", type=int, default=None,
                      help="task id to be run")
  parser.add_argument("--allow-errors", action='store_true',
                      help="continue running next tasks on error (local/srun)")
  args = parser.parse_args()
  return args


def init(task, args_ranges):
  """Sets up the task and argument ranges"""
  global _task_fun
  global _task_args_ranges
  global _num_tasks
  _task_fun = task
  _task_args_ranges = args_ranges
  _num_tasks = num_ids_from_args(_task_args_ranges)

def run_tasks(args):
  if args.action == "local":
    logging.info('Running algorithms locally...')
    _run_local(task_ids=args.tasks, allow_errors=args.allow_errors)
  elif args.action == "srun":
    logging.info('Running algorithms sequentially using srun...')
    _run_srun(task_ids=args.tasks, allow_errors=args.allow_errors)
  elif args.action == "batch":
    logging.info('Running algorithms as batch jobs using Slurm...')
    batch_args = []
    if args.wait:
      batch_args.append("--wait")
    if args.depend:
      args.sbatch_depend = "afterok:" + args.depend
    if args.sbatch_depend:
      batch_args.append("--depend=" + args.sbatch_depend)
    _run_batch(batch_args, task_ids=args.tasks)
  else:
    logging.error("Invalid action '%s'", args.action)


def main():
  """Parses arguments and runs locally, batch or specific task"""
  args = parse_args()
  if args.action == "task":
    assert args.task is not None
    id = int(args.task)
    logging.info('Running task id %d...', id)
    _run_task(id)
  else:
    run_tasks(args)

