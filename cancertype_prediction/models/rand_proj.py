# Gaussian random projection

import numpy as np
#from sklearn.random_projection import GaussianRandomProjection
import scipy.linalg as linalg
from types import SimpleNamespace
import pickle


class RandomProjection:
  def init(self, input_dim, output_dim):
    #self.rp = GaussianRandomProjection(n_components=output_dim)
    self.params = SimpleNamespace()
    self.params.input_dim = input_dim
    self.params.output_dim = output_dim
    return self

  def learn(self, x,
            validation_split=0.0, # unused
            validation_data=None, # unused
            log_file_prefix=None, # unused
            per_epoch_callback_funs=[],
            callbacks=[], # unused
            deadline=None, # unused
            max_duration=None): # unused
    # validation_split not (yet?) supported
    assert validation_split == 0.0
    assert x.shape[1] == self.params.input_dim
    #self.rp.fit(x)
    self.A = np.random.randn(self.params.input_dim,
                             self.params.output_dim)
    self.A = linalg.orth(self.A)
    self.mean = np.mean(x, axis=0)
    for callback in per_epoch_callback_funs:
      callback()
    
  def encode(self, x):
    #return self.rp.transform(x)
    return np.dot(x - self.mean, self.A)

  def decode(self, x):
    #np.linalg.pinv(a)
    #return self.rp.inverse_transform(x)
    return np.dot(x, self.A.T) + self.mean
  
  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.params, f)
      pickle.dump(self.A, f)
      pickle.dump(self.mean, f)
  
  def load (self, filename):
    with open(filename, 'rb') as f:
      self.params = pickle.load(f)
      self.A = pickle.load(f)
      self.mean = pickle.load(f)
    return self

