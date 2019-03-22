"""Just the encoder without decoder
"""

from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.callbacks
import numpy as np

from common import auto_expand
from .nncommon import WeightLogger, LossLogger

class Encoder:
  def init(self,
           input_dim,
           enc_dims,
           output_dim,
           enc_activations,
           n_epochs,
           batch_size,
           dropout=0,
           batch_normalization=False,
           init_pca=False,
           optimizer='adadelta',
           loss='mean_squared_error'):
    self.input_dim = input_dim
    self.output_dim = output_dim

    # validate and deduce some parameters
    assert isinstance(enc_activations, str) or len(enc_activations) == len(enc_dims) + 1 
    n_enc = len(enc_dims)

    # encoding layers
    self.enc_layers = []
    enc_dims.append(output_dim)
    for i in range(n_enc + 1):
      if dropout > 0:
        self.enc_layers.append(Dropout(dropout))
      self.enc_layers.append(Dense(enc_dims[i]))
      if i == 0:
        self.first_dense_enc_layer = self.enc_layers[-1]
      if batch_normalization:
        self.enc_layers.append(BatchNormalization(mode=2))
      self.enc_layers.append(Activation(auto_expand(enc_activations)[i]))
    self.enc_layers.append(Activation(None, name="encoded"))
    
    #import pdb; pdb.set_trace()
    # input placeholder
    self.input = Input(shape=(input_dim,))

    # create encoder
    self.encoded = self.input
    for layer in self.enc_layers:
      self.encoded = layer(self.encoded)
    self.encoder = Model(input=self.input, output=self.encoded)

    #from keras.utils.visualize_util import plot
    #plot(self.autoencoder, to_file='ae.png')

    # learning parameters
    self.init_pca = init_pca
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.loss = loss

    return self

  def learn(self, x, y, log_file_prefix=None, callbacks=[]):
    # define optimizer and loss function
    self.encoder.compile(optimizer=self.optimizer, loss=self.loss)

    if (self.init_pca):
      # initialize weights of the first and last layer using PCA 
      from sklearn.decomposition import PCA as sk_PCA
      #weights = self.encoder.layers[1].get_weights()
      weights = self.first_dense_enc_layer.get_weights()
      dim = weights[1].size
      w = sk_PCA(n_components=dim).fit(x).components_
      weights[0][:,:] = w.T
      weights[1][:] = -np.mean(np.dot(x, w.T), axis=0)
      #self.encoder.layers[1].set_weights(weights)
      self.first_dense_enc_layer.set_weights(weights)
    
    # optionally add callbacks
    keras_callbacks = []
    if log_file_prefix:
      #keras_callbacks.append(keras.callbacks.CSVLogger(log_file_prefix + ".log"))
      keras_callbacks.append(WeightLogger(self.encoder, log_file_prefix))
      keras_callbacks.append(LossLogger(log_file_prefix))
    for callback in callbacks:
      class CB(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
          callback()
        def on_epoch_end(self, epoch, logs={}):
          callback()
      keras_callbacks.append(CB())
    
    # train
    self.encoder.fit(x, y,
        nb_epoch=self.n_epochs,
        batch_size=self.batch_size,
        shuffle=True,
        callbacks=keras_callbacks,
        verbose=2
        )
    
  def encode(self, x):
      return self.encoder.predict(x)

  def save(self, filename):
    self.encoder.save_weights(filename)

  def load(self, filename):
    self.encoder.load_weights(filename)
