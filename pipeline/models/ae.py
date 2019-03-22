"""Traditional autoencoder
"""

from types import SimpleNamespace
import numpy as np
import pickle

from keras.layers import Input, Dense, Dropout, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.callbacks
from keras import backend as K

from common import auto_expand
from .nncommon import WeightLogger, WeightDiffStatLogger, LossLogger, create_activation

class AE:
  def init(self,
           input_dim,
           enc_dims,
           output_dim,
           dec_dims,
           enc_activations,
           dec_activations,
           n_epochs,
           batch_size,
           reconstruction_dim=None,
           secondary_dims = None,
           secondary_activations = None,
           normalize_input_mean = None,
           normalize_input_scale = None,
           dropout=0,
           secondary_dropout=0,
           batch_normalization=False,
           init_pca=False,
           pre_train=None,
           optimizer='SGD',
           loss='mean_squared_error',
           secondary_loss='mean_squared_error',
           secondary_loss_weight=0.5,
           early_stopping=False,
           early_stopping_patience=10,
           log_weights=True,
           log_weights_diff_norm=None,
           log_loss=True,
           log_loss_per_patch=False):
    
    self.params = SimpleNamespace()

    if reconstruction_dim is None:
      reconstruction_dim = input_dim
    self.params.input_dim = input_dim
    self.params.reconstruction_dim = reconstruction_dim
    self.params.output_dim = output_dim
    self.params.secondary_dims = secondary_dims
    self.params.secondary_activations = secondary_activations
    self.params.normalize_input_mean = normalize_input_mean
    self.params.normalize_input_scale = normalize_input_scale

    # validate and deduce some parameters
    assert (not isinstance(enc_activations, list)
            or len(enc_activations) == len(enc_dims) + 1)
    self.params.enc_activations = enc_activations
    self.params.enc_dims = enc_dims
    self.params.n_enc = len(enc_dims)
    if dec_dims == "same":
      dec_dims = list(reversed(enc_dims))
    self.params.dec_dims = dec_dims
    assert (not isinstance(dec_activations, list)
            or len(dec_activations) == len(dec_dims) + 1)
    self.params.dec_activations = dec_activations
    self.params.n_dec = len(dec_dims)
    if secondary_dims is not None:
      self.params.n_secondary = len(secondary_dims)

    # learning parameters
    self.params.n_epochs = n_epochs
    self.params.batch_size = batch_size
    self.params.dropout = dropout
    self.params.secondary_dropout = secondary_dropout
    self.params.batch_normalization = batch_normalization
    self.params.init_pca = init_pca
    self.params.pre_train = pre_train
    self.params.optimizer = optimizer
    self.params.loss = loss
    self.params.secondary_loss = secondary_loss
    self.params.secondary_loss_weight = secondary_loss_weight
    self.params.early_stopping = early_stopping
    self.params.early_stopping_patience = early_stopping_patience
    self.params.log_weights = log_weights
    self.params.log_weights_diff_norm = log_weights_diff_norm
    self.params.log_loss = log_loss
    self.log_loss_per_patch = log_loss_per_patch

    self.build_models()

    return self

  def build_models(self):
    # encoding layers
    self.enc_layers = []
    enc_dims = list(self.params.enc_dims)
    enc_dims.append(self.params.output_dim)
    enc_activations = auto_expand(self.params.enc_activations)
    for i in range(self.params.n_enc + 1):
      if self.params.dropout:
        self.enc_layers.append(Dropout(auto_expand(self.params.dropout)[i]))
      add_bias_terms = not self.params.batch_normalization
      self.enc_layers.append(Dense(enc_dims[i], bias=add_bias_terms))
      if i == 0:
        self.first_dense_enc_layer = self.enc_layers[-1]
      if self.params.batch_normalization:
        self.enc_layers.append(BatchNormalization(mode=2))
      self.enc_layers.append(create_activation(enc_activations[i]))
    
    # decoding layers
    self.dec_layers = []
    dec_dims = list(self.params.dec_dims)
    dec_dims.append(self.params.reconstruction_dim)
    dec_activations = auto_expand(self.params.dec_activations)
    for i in range(self.params.n_dec + 1):
      add_bias_terms = not (self.params.batch_normalization and i != self.params.n_dec)
      self.dec_layers.append(Dense(dec_dims[i], bias=add_bias_terms))
      if i == self.params.n_dec:
        self.last_dense_dec_layer = self.dec_layers[-1]
      if self.params.batch_normalization and i != self.params.n_dec:
        self.dec_layers.append(BatchNormalization(mode=2))
      self.dec_layers.append(create_activation(dec_activations[i]))

    # normalization layers
    if self.params.normalize_input_scale is not None:
      scale = K.variable(self.params.normalize_input_scale)
      self.enc_layers.insert(0, Lambda(lambda x:
           x / scale))
      self.dec_layers.append(Lambda(lambda x:
           x * scale))
    if self.params.normalize_input_mean is not None:
      mean = K.variable(self.params.normalize_input_mean)
      self.enc_layers.insert(0, Lambda(lambda x:
           x - mean))
      self.dec_layers.append(Lambda(lambda x:
           x + mean))
    
    # named layers
    self.enc_layers.append(Activation(None, name="encoded"))
    self.dec_layers.append(Activation(None, name="decoded"))

    # secondary layers
    if self.params.secondary_dims is not None:
      self.sec_layers = []
      sec_dims = list(self.params.secondary_dims)
      sec_activations = auto_expand(self.params.secondary_activations)
      for i in range(self.params.n_secondary):
        if self.params.secondary_dropout:
          self.enc_layers.append(Dropout(auto_expand(self.params.secondary_dropout)[i]))
        add_bias_terms = not (self.params.batch_normalization and
                              i != self.params.n_secondary - 1)
        self.sec_layers.append(Dense(sec_dims[i], bias=add_bias_terms))
        if self.params.batch_normalization and i != self.params.n_secondary - 1:
          self.sec_layers.append(BatchNormalization(mode=2))
        self.sec_layers.append(create_activation(sec_activations[i]))
      self.sec_layers.append(Activation(None, name="secondary"))
    
    #import pdb; pdb.set_trace()
    # input placeholder
    self.input = Input(shape=(self.params.input_dim,))

    # create encoder
    self.encoded = self.input
    for layer in self.enc_layers:
      self.encoded = layer(self.encoded)
    self.encoder = Model(input=self.input, output=self.encoded)

    # create autoencoder and secondary predictor
    self.encoded_and_decoded = self.encoded
    for layer in self.dec_layers:
      self.encoded_and_decoded = layer(self.encoded_and_decoded)
    if self.params.secondary_dims is None:
      self.autoencoder = Model(input=self.input, output=self.encoded_and_decoded)
    else:
      self.secondary = self.encoded
      for layer in self.sec_layers:
        self.secondary = layer(self.secondary)
      self.autoencoder = Model(input=self.input,
                               output=[self.encoded_and_decoded, self.secondary])
      self.secondary_predicter = Model(input=self.input,
                                       output=self.secondary)

    # create decoder
    self.encoded_input = Input(shape=(self.params.output_dim,))
    self.decoded = self.encoded_input
    for layer in self.dec_layers:
      self.decoded = layer(self.decoded)
    self.decoder = Model(input=self.encoded_input, output=self.decoded)
  
  
  def learn(self, x,
            y=None,
            secondary_y=None,
            validation_split=0.0,
            validation_data=None,
            secondary_validation_data=None,
            log_file_prefix=None,
            per_epoch_callback_funs=[],
            callbacks=[]):
    
    # define optimizer and loss function
    if secondary_y is None:
      loss = self.params.loss
      loss_weights = None
    else:
      assert self.params.secondary_dims is not None
      loss = [self.params.loss, self.params.secondary_loss]
      loss_weights = [1 - self.params.secondary_loss_weight,
                      self.params.secondary_loss_weight]
    self.autoencoder.compile(optimizer=self.params.optimizer, loss=loss,
                             loss_weights=loss_weights)

    if (self.params.init_pca):
      # initialize weights of the first and last layer using PCA 
      from sklearn.decomposition import PCA as sk_PCA
      #weights = self.autoencoder.layers[1].get_weights()
      weights = self.first_dense_enc_layer.get_weights()
      dim = weights[1].size
      w = sk_PCA(n_components=dim).fit(x).components_
      weights[0][:,:] = w.T
      weights[1][:] = -np.mean(np.dot(x, w.T), axis=0)
      #self.autoencoder.layers[1].set_weights(weights)
      self.first_dense_enc_layer.set_weights(weights)
      #weights = self.autoencoder.layers[-1].get_weights()
      weights = self.last_dense_dec_layer.get_weights()
      weights[0][:,:] = w
      weights[1][:] = np.mean(x, axis=0)
      #self.autoencoder.layers[-1].set_weights(weights)
      self.last_dense_dec_layer.set_weights(weights)
    
    # possible validation data
    if validation_data is not None and y is None:
      if secondary_validation_data is not None:
        validation_data = (validation_data,
                           [validation_data, secondary_validation_data])
      else:
        validation_data = (validation_data, validation_data)
  
    validation = (validation_data is not None or validation_split > 0)
    
    # by default predict the data itself
    if y is None:
      y = x
    
    if secondary_y is not None:
      y = [y, secondary_y]

    # optionally add callbacks
    keras_callbacks = []
    # 'built-in' callbacks
    if log_file_prefix:
      #keras_callbacks.append(keras.callbacks.CSVLogger(log_file_prefix + ".log"))
      if self.params.log_weights:
        keras_callbacks.append(WeightLogger(self.autoencoder, log_file_prefix))
      if self.params.log_weights_diff_norm is not None:
        keras_callbacks.append(WeightDiffStatLogger(self.autoencoder, log_file_prefix, self.params.log_weights_diff_norm))
      if self.params.log_loss:
        keras_callbacks.append(LossLogger(log_file_prefix,
                                          per_patch=self.log_loss_per_patch))
        if secondary_y is not None:
          keras_callbacks.append(LossLogger(log_file_prefix,
                                            loss='decoded_loss',
                                            per_patch=self.log_loss_per_patch))
          keras_callbacks.append(LossLogger(log_file_prefix,
                                            loss='secondary_loss',
                                            per_patch=self.log_loss_per_patch))
      if self.params.log_loss and validation:
        keras_callbacks.append(LossLogger(log_file_prefix, loss='val_loss'))
        if secondary_y is not None:
          keras_callbacks.append(LossLogger(log_file_prefix,
                                            loss='val_decoded_loss'))
          keras_callbacks.append(LossLogger(log_file_prefix,
                                            loss='val_secondary_loss'))
    # externally defined keras callback objects
    for callback in callbacks:
      keras_callbacks.extend(callbacks)
    # externally defined callbacks functions
    for callback in per_epoch_callback_funs:
      class CB(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
          callback()
        def on_epoch_end(self, epoch, logs={}):
          callback()
      keras_callbacks.append(CB())
    
    if self.params.early_stopping:
      if self.params.early_stopping == True:
        monitor = ('val_loss' if validation else 'loss')
      else:
        monitor = self.params.early_stopping
      keras_callbacks.append(keras.callbacks.EarlyStopping(
          monitor=monitor,
          patience=self.params.early_stopping_patience))
    
    # optional pre train
    if self.params.pre_train is not None:
      method, params = self.params.pre_train
      if method == "pca":
        # fit encoder and decoder separately to PCA output
        pre_n_epochs, = params
        from sklearn.decomposition import PCA as sk_PCA
        y = sk_PCA(n_components=self.params.output_dim).fit_transform(x)
        pretrain_combined = True
        if pretrain_combined:
          # (pre)train both encoder and decoder at the same time
          encoder_and_decoder = Model(input=[self.input, self.encoded_input],
                                      output=[self.encoded, self.decoded])
          encoder_and_decoder.compile(optimizer=self.params.optimizer, loss=self.params.loss)
          pretrain_keras_callbacks = keras_callbacks.copy()
          if log_file_prefix:
            pretrain_keras_callbacks.append(LossLogger(log_file_prefix, loss='encoded_loss'))
            pretrain_keras_callbacks.append(LossLogger(log_file_prefix, loss='decoded_loss'))
          encoder_and_decoder.fit([x, y], [y, x],
              nb_epoch=pre_n_epochs,
              batch_size=self.params.batch_size,
              shuffle=True,
              callbacks=pretrain_keras_callbacks,
              verbose=2
              )
        else:
          self.encoder.compile(optimizer=self.params.optimizer, loss=self.params.loss)
          self.encoder.fit(x, y,
              nb_epoch=pre_n_epochs,
              batch_size=self.params.batch_size,
              shuffle=True,
              callbacks=keras_callbacks,
              verbose=2
              )
          self.decoder.compile(optimizer=self.params.optimizer, loss=self.params.loss)
          self.decoder.fit(y, x,
              nb_epoch=pre_n_epochs,
              batch_size=self.params.batch_size,
              shuffle=True,
              callbacks=keras_callbacks,
              verbose=2
              )
      else:
        raise ValueError("Invalid pre train method '%s'" % method)

    # train
    self.autoencoder.fit(x, y,
        nb_epoch=self.params.n_epochs,
        batch_size=self.params.batch_size,
        shuffle=True,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=keras_callbacks,
        verbose=2
        )
    
  def encode(self, x):
    return self.encoder.predict(x)

  def decode(self, x):
    return self.decoder.predict(x)
  
  def predict_secondary(self, x):
    return self.secondary_predicter.predict(x)

  def save(self, filename):
    with open(filename + "_params.pkl", 'wb') as f:
      pickle.dump(self.params, f)
    self.autoencoder.save_weights(filename + "_weights.h5")
    #keras.models.save_model(self.autoencoder, filename + "_autoencoder.h5")
    #keras.models.save_model(self.encoder, filename + "_encoder.h5")
    #keras.models.save_model(self.decoder, filename + "_decoder.h5")

  def load(self, filename):
    with open(filename + "_params.pkl", 'rb') as f:
      self.params = pickle.load(f)
    self.build_models()
    self.autoencoder.load_weights(filename + "_weights.h5")
    #self.autoencoder = keras.models.load_model(filename + "_autoencoder.h5")
    #self.encoder = keras.models.load_model(filename + "_encoder.h5")
    #self.decoder = keras.models.load_model(filename + "_decoder.h5")
    return self
  
  def get_encoder(self):
    return self.encoder
