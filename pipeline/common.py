"""Common utility module
"""

def ensure_dir_exists(path):
  """Make sure that the given directory exists"""
  import os
  import errno
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

def auto_expand(list_or_value):
  """Given a list return it, given a scalar return a corresponding broadcasted 'infinite list'"""
  if isinstance(list_or_value, list):
    return list_or_value
  else:
    class Expanded:
      def __init__(self, value):
        self.value = value
      def __getitem__(self, i):
        return self.value
    return Expanded(list_or_value)


def expInterp(a, b, x):
  """Interpolate exponentially"""
  import numpy as np
  return np.exp(x * np.log(b) + (1 - x) * np.log(a))

def args_from_id(id, arg_ranges):
  """Convert a single id number to the corresponding set of arguments"""
  args = list()
  for arg_range in arg_ranges:
    i = id % len(arg_range)
    id = id // len(arg_range)
    args.append(arg_range[i])
  return tuple(args)

def num_ids_from_args(arg_ranges):
  """Return the number of argument combinations (and thus the number of corresponding ids)"""
  from functools import reduce
  import operator
  return reduce(operator.mul, [len(r) for r in arg_ranges], 1)

def print_err(*args, **kwargs):
  import sys
  print(*args, file=sys.stderr, **kwargs)

