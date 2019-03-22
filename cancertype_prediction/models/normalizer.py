# Gaussian random projection

import numpy as np
from types import SimpleNamespace
import pickle


class Normalizer:
  def __init__(self):
    self.offset = 0.0
    self.scale = 1.0

  def init(self, **kwargs):
    #self.params = params
    self.params = SimpleNamespace(**kwargs)
    return self

  def learn(self, x):
    if self.params.normalize_input_axis == 'global':
      axis = None
    elif self.params.normalize_input_axis == 'feature_wise':
      axis = 0
    else:
      assert False, "invalid normalize_input_axis"

    if self.params.normalize_input_type is None:
      self.offset = 0.0
      self.scale = 1.0
    elif self.params.normalize_input_type == 'mean':
      self.offset = np.mean(x, axis=axis) - self.params.normalize_input_target
      self.scale = 1.0
    elif self.params.normalize_input_type == 'median':
      self.offset = np.median(x, axis=axis) - self.params.normalize_input_target
      self.scale = 1.0
    elif self.params.normalize_input_type == 'min':
      self.offset = np.amin(x, axis=axis) - self.params.normalize_input_target
      self.scale = 1.0
    elif self.params.normalize_input_type == 'max':
      self.offset = np.amax(x, axis=axis) - self.params.normalize_input_target
      self.scale = 1.0
    elif self.params.normalize_input_type == 'stddev':
      m = np.mean(x, axis=axis)
      radius = np.std(x - m, axis=axis) * self.params.normalize_input_stddev_mult
      (l1, r1) = self.params.normalize_input_target
      self.scale = 2 * radius / (r1 - l1)
      self.offset = m - radius - l1 * self.scale
    elif self.params.normalize_input_type == 'quantiles':
      q = self.params.normalize_input_quantile
      l0 = np.percentile(x, q * 100, axis=axis)
      r0 = np.percentile(x, (1 - q) * 100, axis=axis)
      (l1, r1) = self.params.normalize_input_target
      self.scale = (r0 - l0) / (r1 - l1)
      self.offset = l0 - l1 * self.scale
    elif self.params.normalize_input_type == 'minmax':
      l0 = np.amin(x, axis=axis)
      r0 = np.amax(x, axis=axis)
      (l1, r1) = self.params.normalize_input_target
      self.scale = (r0 - l0) / (r1 - l1)
      self.offset = l0 - l1 * self.scale
    else:
      assert False, "invalid normalize_input_type"
    
  def normalize(self, x):
    x = (x - self.offset) / self.scale
    if self.params.normalize_input_clip:
      x = np.clip(x, *(self.params.normalize_input_target))
    return x

  def restore(self, x):
    return x * self.scale + self.offset
  
  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.params, f)
      pickle.dump(self.offset, f)
      pickle.dump(self.scale, f)
  
  def load (self, filename):
    with open(filename, 'rb') as f:
      self.params = pickle.load(f)
      self.offset = pickle.load(f)
      self.scale = pickle.load(f)
    return self

