# PCA

import numpy as np
from sklearn.decomposition import PCA as sk_PCA
import pickle


class PCA:
  def init(self, input_dim, output_dim):
    self.pca = sk_PCA(n_components=output_dim)
    return self

  def learn(self, x,
            validation_split=0.0, # unused
            validation_data=None, # unused
            log_file_prefix=None, # unused
            per_epoch_callback_funs=[],
            callbacks=[]): # unused
    # validation_split not (yet?) supported
    assert validation_split == 0.0
    self.pca.fit(x)
    for callback in per_epoch_callback_funs:
      callback()
    
  def encode(self, x):
    return self.pca.transform(x)

  def decode(self, x):
    return self.pca.inverse_transform(x)
  
  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.pca, f)
  
  def load (self, filename):
    with open(filename, 'rb') as f:
      self.pca = pickle.load(f)
    return self

