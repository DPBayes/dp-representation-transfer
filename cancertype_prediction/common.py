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

def pretty_duration(dur):
  import datetime
  if isinstance(dur, datetime.timedelta):
    s = dur.total_seconds()
  else:
    s = dur
  m, s = divmod(s, 60)
  out = "%g s" % (s)
  if m > 0:
    h, m = divmod(m, 60)
    out = "%d min " % (m) + out
    if h > 0:
      d, h = divmod(h, 24)
      out = "%d h " % (h) + out
      if d > 0:
        out = "%d d " % (d) + out
  return out


def categorical_to_binary(y):
  """Converts a class vector (pandas categorical) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  # Arguments
      y: class vector to be converted into a matrix
          (pandas.core.categorical.Categorical).
  # Returns
      A binary matrix representation of the input.
  """
  num_classes = y.categories.size
  avail = np.where(y.codes != -1)[0]
  not_avail = np.where(y.codes == -1)[0]
  n = y.size
  bin_matrix = np.zeros((n, num_classes))
  bin_matrix[avail, y.codes[avail]] = 1
  bin_matrix[not_avail, :] = 1 / num_classes
  return bin_matrix
