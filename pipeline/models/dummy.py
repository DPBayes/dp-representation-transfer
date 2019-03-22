"""Dummy model that encodes everything to zero and decodes everything its mean
"""

import numpy as np

class Dummy:
  def init(self, input_dim, output_dim):
    self.output_dim = output_dim
    return self

  def learn(self, x, log_file_prefix=None, callbacks=[]):
    self.average = np.average(x, axis=0)
    for callback in callbacks:
      callback()
    
  def encode(self, x):
    return np.zeros((x.shape[0], self.output_dim))

  def decode(self, x):
    return np.broadcast_to(self.average, (x.shape[0], self.average.size))
    #return np.tile(self.average, (x.shape[0]))
    #return np.zeros((x.shape[0], self.input_dim))
  
  def save(self, filename):
    pass
  
  def load (self, filename):
    return self

