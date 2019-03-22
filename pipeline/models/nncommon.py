"""Common/misc. functionality related to neural networks / Keras
"""

import keras.callbacks
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
import numpy as np
import datetime

class WeightLogger(keras.callbacks.Callback):
  def __init__(self, model, filename_prefix):
    self.model = model
    self.filename_prefix = filename_prefix
    self.start()
  
  def write(self):
    for layer in self.model.layers:
      weights = layer.get_weights()
      for w in weights:
        self.weight_file.write(" ")
        w.tofile(self.weight_file, sep=' ', format="%.5e")
    self.weight_file.write("\n")
  
  def start(self):
    import pickle
    # output layer names
    layer_names = [layer.name for layer in self.model.layers]
    with open(self.filename_prefix + "-weight-layer-names.p", 'wb') as f:
      pickle.dump(layer_names, f)
    # output weight shapes
    w_shapes = [[w.shape for w in layer.get_weights()] for layer in self.model.layers]
    with open(self.filename_prefix + "-weight-shapes.p", 'wb') as f:
      pickle.dump(w_shapes, f)
    # open then weight file and write the initial state
    self.weight_file = open(self.filename_prefix + "-weights.txt", 'w', encoding='utf-8')
    self.write()
  
  #def on_train_begin(self, logs={}):
  #  self.start()

  def on_epoch_end(self, epoch, logs={}):
    self.write()
  
  #def on_train_end(self, logs={}):
  #  self.weight_file.close()

class WeightDiffStatLogger(keras.callbacks.Callback):
  def __init__(self, model, filename_prefix, norm_ord=2):
    self.model = model
    self.filename_prefix = filename_prefix
    self.norm_ord = norm_ord
    self.start()
  
  def get_weights(self):
    return np.concatenate([w.flatten() for w in self.model.get_weights()])

  def start(self):
    self.norm_file = open(self.filename_prefix + "-weight-diff-norm-perpatch.txt",
                         'w', encoding='utf-8')
    self.dir_coeff_file = open(self.filename_prefix + "-weight-diff-dir-coeff-perpatch.txt",
                         'w', encoding='utf-8')
    #self.prev_weights = K.batch_get_value(self.model.trainable_weights))
    self.prev_weights = self.get_weights()
    self.prev_diff = self.prev_weights - self.prev_weights
  
  #def on_train_begin(self, logs={}):
  #  self.start()

  def write(self):
    #weights = K.batch_get_value(self.model.trainable_weights)
    weights = self.get_weights()
    diff = weights - self.prev_weights
    norm = np.linalg.norm(diff, ord=self.norm_ord)
    dir_coeff = (np.dot(diff, self.prev_diff) /
                 (np.linalg.norm(diff) * np.linalg.norm(self.prev_diff)))
    self.norm_file.write(" %.5e" % norm)
    self.dir_coeff_file.write(" %.5e" % dir_coeff)
    self.prev_weights = weights
    self.prev_diff = diff

  #def on_epoch_end(self, epoch, logs={}):
  #  self.write()

  def on_batch_end(self, batch, logs={}):
    self.write()
  
  #def on_train_end(self, logs={}):
  #  self.out_file.close()


class LossLogger(keras.callbacks.Callback):
  def __init__(self, filename_prefix, loss='loss', per_patch=False):
    self.filename_prefix = filename_prefix
    self.loss = loss
    self.per_patch = per_patch
    self.start()

  def start(self):
    self.loss_file = open("%s-%s.txt" % (self.filename_prefix, self.loss),
                          'w', encoding='utf-8')
    if self.per_patch:
      self.perpatch_loss_file = open("%s-%s-perpatch.txt" % (self.filename_prefix, self.loss),
                            'w', encoding='utf-8')
    #self.loss_file.write("%.5e" % logs.get(loss))

  #def on_train_begin(self, logs={}):
  #  self.start()

  def on_epoch_end(self, epoch, logs={}):
    self.loss_file.write(" %.5e" % logs.get(self.loss))

  def on_batch_end(self, batch, logs={}):
    if self.per_patch:
      self.perpatch_loss_file.write(" %.5e" % logs.get(self.loss))

  #def on_train_end(self, logs={}):
  #  self.loss_file.close()



class TimeBasedStopping(keras.callbacks.Callback):
  """Stop training when given time has elapsed.
  """

  def __init__(self, max_duration=None, deadline=None, verbose=0):
    super(TimeBasedStopping, self).__init__()
    if max_duration is None and deadline is None:
      raise ValueError("Either max_duration or deadline must be provided")
    self.max_duration = max_duration
    self.deadline = deadline
    self.verbose = verbose
    self.stopped_epoch = None

  def on_train_begin(self, logs=None):
    if self.max_duration is not None:
      new_dl = datetime.datetime.now() + self.max_duration
      if self.deadline is None or new_dl < self.deadline:
        self.deadline = new_dl

  def on_epoch_end(self, epoch, logs=None):
    if datetime.datetime.now() >= self.deadline:
      self.model.stop_training = True
      self.stopped_epoch = epoch

  def on_train_end(self, logs=None):
    if self.stopped_epoch is not None and self.verbose > 0:
      print('Epoch %d: time based stopping' % (self.stopped_epoch))



def create_activation(name):
  if name.lower() == 'LeakyReLU'.lower():
    return LeakyReLU()
  elif name.lower() == 'PReLU'.lower():
    return PReLU()
  elif name.lower() == 'ELU'.lower():
    return ELU()
  else:
    return Activation(name)

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