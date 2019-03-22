'''
Variational auto-encoder.
'''

from types import SimpleNamespace
import numpy as np
import pickle

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import optimizers
import keras.callbacks
from keras.optimizers import *
from keras import initializations
from keras.engine.topology import Layer

from common import auto_expand
from .nncommon import WeightLogger, WeightDiffStatLogger, LossLogger, create_activation

class Gaussian(Layer):
  '''Generate Gaussian output. First input defines the mean and second defines the log of variance.
  '''
  def __init__(self, **kwargs):
    super(Gaussian, self).__init__(**kwargs)
  def call(self, args, mask=None):
    mean, log_var = args
    #assert K.shape(mean) == K.shape(log_var)
    z_standard = K.random_normal(shape=K.shape(mean), mean=0.0, std=1.0)
    return mean + K.exp(log_var / 2) * z_standard
    #return K.in_train_phase(noise_x, x)
  def get_output_shape_for(self, input_shape):
    return input_shape[0]

class Const(Layer):
  '''Layer with constant output (i.e. independent of the input). The output values can be trained.
  Requires some input to make Keras happy.'''
  def __init__(self, shape=None, trainable=False, init='zero', value=None, **kwargs):
    super(Const, self).__init__(**kwargs)
    assert trainable or (value is not None)
    self.init = initializations.get(init)
    self.initial_value = value
    self.trainable = trainable
    self.shape = shape
  def build(self, input_shape):
    #assert len(input_shape) == 0
    if self.trainable:
      self.value = self.init(self.shape, '{}_value'.format(self.name))
      #self.log_var = self.add_weight(name='log_var', shape=(log_var_dim,), initializer='zeros', trainable=True)
      self.trainable_weights = [self.value]
    if self.initial_value is not None:
      if self.trainable:
        self.set_weights(self.initial_value)
      else:
        self.value = np.array(self.initial_value)
        self.shape = self.value.shape
    super(Const, self).build(input_shape)
  def call(self, args, mask=None):
    #assert len(args) == 0
    #return self.value
    return self.value + 0 * args[0,0] #K.sum(args, axis=1)
  def get_output_shape_for(self, input_shape):
    #return (1,) + self.shape
    return (input_shape[0],) + self.shape

class GaussianLogProb(Layer):
  '''Log probability of Gaussian distribution.'''
  def __init__(self, **kwargs):
    super(GaussianLogProb, self).__init__(**kwargs)
  def call(self, args, mask=None):
    x, mean, log_var = args
    return K.sum(- K.square(x - mean) / (2 * K.exp(log_var)) -
                 0.5 * (K.log(2 * np.pi) + log_var), axis=1)
  def get_output_shape_for(self, input_shape):
    return (input_shape[0][0], 1)

class StdGaussianKLDist(Layer):
  '''Kullback-Leibler divergence between the standard normal distribution and a (non-standard) normal distribution whose parameters are given as input.'''
  def __init__(self, **kwargs):
    super(StdGaussianKLDist, self).__init__(**kwargs)
  def call(self, args, mask=None):
    mean, log_var = args
    return -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=1)
  def get_output_shape_for(self, input_shape):
    return (input_shape[0][0], 1)


class VAELoss(Layer):
  '''Loss for variational autoencoder. Inputs: KL distance, log likelihood'''
  def __init__(self, **kwargs):
    super(VAELoss, self).__init__(**kwargs)
  def call(self, args, mask=None):
    kl_dist, log_prob = args
    return kl_dist - log_prob
    #return -log_prob
  def get_output_shape_for(self, input_shape):
    return input_shape[0]

class SumLoss(Layer):
  '''Total loss that is a sum of input losses'''
  def __init__(self, **kwargs):
    super(SumLoss, self).__init__(**kwargs)
  def call(self, args, mask=None):
    tot_loss = 0
    for loss in args:
      tot_loss += loss
    return tot_loss
    #return reduce(lambda x, y: x + y, args)
  def get_output_shape_for(self, input_shape):
    return input_shape[0]

def pass_through_loss(dummy, loss):
  return loss

class VAE:
  def init(self,
           input_dim,
           latent_dim,
           enc_dims,
           dec_dims,
           enc_activations,
           dec_activations,
           prediction_var='global_same', #one of: 'persample_independent'
                                         #        'persample_same'
                                         #        'global_independent'
                                         #        'global_same'
                                         #        constant scalar
                                         #        constant vector
           prediction_log_var_min=None,
           n_epochs=None,
           batch_size=64,
           #normalize_input_mean = None,
           #normalize_input_scale = None,
           #dropout=0,
           input_dropout=0,
           enc_dropout=0,
           latent_dropout=0,
           dec_dropout=0,
           optimizer='Adam',
           optimizer_params={},
           early_stopping=False,
           early_stopping_patience=10,
           reduce_lr_on_plateau=False,
           reduce_lr_factor=0.1,
           reduce_lr_patience=5,
           log_weights=True,
           log_loss=True,
           log_loss_per_patch=False):
    
    self.params = SimpleNamespace()

    self.params.input_dim = input_dim
    self.params.latent_dim = latent_dim
    self.params.prediction_var = prediction_var
    self.params.prediction_log_var_min = prediction_log_var_min
    #self.params.normalize_input_mean = normalize_input_mean
    #self.params.normalize_input_scale = normalize_input_scale

    assert (not isinstance(enc_activations, list)
            or len(enc_activations) == len(enc_dims))
    self.params.enc_activations = enc_activations
    self.params.enc_dims = enc_dims
    self.params.n_enc = len(enc_dims)
    if dec_dims == "same":
      dec_dims = list(reversed(enc_dims))
    self.params.dec_dims = dec_dims
    assert (not isinstance(dec_activations, list)
            or len(dec_activations) == len(dec_dims))
    self.params.dec_activations = dec_activations
    self.params.n_dec = len(dec_dims)

    # learning parameters
    self.params.n_epochs = n_epochs
    self.params.batch_size = batch_size
    #self.params.dropout = dropout
    self.params.input_dropout = input_dropout
    self.params.enc_dropout = enc_dropout
    self.params.latent_dropout = latent_dropout
    self.params.dec_dropout = dec_dropout
    #self.params.batch_normalization = batch_normalization
    self.params.optimizer = optimizer
    self.params.optimizer_params = optimizer_params
    self.params.early_stopping = early_stopping
    self.params.early_stopping_patience = early_stopping_patience
    self.reduce_lr_on_plateau = reduce_lr_on_plateau
    self.reduce_lr_factor = reduce_lr_factor
    self.reduce_lr_patience = reduce_lr_patience
    self.params.log_weights = log_weights
    self.params.log_loss = log_loss
    self.log_loss_per_patch = log_loss_per_patch

    self.build_models()

    return self


  def build_models(self):
    # encoding layers
    self.enc_layers = []
    enc_dims = list(self.params.enc_dims)
    enc_activations = auto_expand(self.params.enc_activations)
    if self.params.input_dropout:
      self.enc_layers.append(Dropout(self.params.input_dropout))
    for i in range(self.params.n_enc):
      #add_bias_terms = not self.params.batch_normalization
      add_bias_terms = True
      self.enc_layers.append(Dense(enc_dims[i], bias=add_bias_terms,
                                   name='enc_hidden_%d'%(i+1)))
      if i == 0:
        self.first_dense_enc_layer = self.enc_layers[-1]
      #if self.params.batch_normalization:
      #  self.enc_layers.append(BatchNormalization(mode=2))
      self.enc_layers.append(create_activation(enc_activations[i]))
      if self.params.enc_dropout:
        self.enc_layers.append(Dropout(auto_expand(
             self.params.enc_dropout)[i]))
    
    # latent layers
    self.enc_z_mean = Dense(self.params.latent_dim, name='latent_mean')
    self.enc_z_log_var = Dense(self.params.latent_dim, name='latent_log_var')
    self.enc_z_draw = Gaussian(name='latent_draw')
    
    # decoding layers
    self.dec_layers = []
    dec_dims = list(self.params.dec_dims)
    dec_activations = auto_expand(self.params.dec_activations)
    if self.params.latent_dropout:
      self.dec_layers.append(Dropout(self.params.latent_dropout))
    for i in range(self.params.n_dec):
      #add_bias_terms = not (self.params.batch_normalization and i != self.params.n_dec)
      add_bias_terms = True
      self.dec_layers.append(Dense(dec_dims[i], bias=add_bias_terms,
                                   name='dec_hidden_%d'%(i+1)))
      if i == self.params.n_dec:
        self.last_dense_dec_layer = self.dec_layers[-1]
      #if self.params.batch_normalization and i != self.params.n_dec:
      #  self.dec_layers.append(BatchNormalization(mode=2))
      self.dec_layers.append(create_activation(dec_activations[i]))
      if self.params.dec_dropout:
        self.dec_layers.append(Dropout(auto_expand(
             self.params.dec_dropout)[i]))
    
    # prediction layers
    self.dec_x_mean = Dense(self.params.input_dim, name='pred_mean')
    pvname = 'pred_log_var'
    if self.params.prediction_var in ['persample_independent', 'pi']:
      self.dec_x_log_var = Dense(self.params.input_dim, init='zero', name=pvname)
    elif self.params.prediction_var in ['persample_same', 'ps']:
      self.dec_x_log_var = Dense(1, init='zero', name=pvname)
    elif self.params.prediction_var in ['global_independent', 'gi']:
      self.dec_x_log_var = Const(shape=(self.params.input_dim,), trainable=True, name=pvname)
    elif self.params.prediction_var in ['global_same', 'gs']:
      self.dec_x_log_var = Const(shape=(1,), trainable=True, name=pvname)
    else:
      self.dec_x_log_var = Const(value=self.params.prediction_var, name=pvname)
    self.dec_x_draw = Gaussian(name='pred_draw')

    # normalization layers
    #if self.params.normalize_input_scale is not None:
    #  scale = K.variable(self.params.normalize_input_scale)
    #  self.enc_layers.insert(0, Lambda(lambda x:
    #       x / scale))
    #  self.dec_layers.append(Lambda(lambda x:
    #       x * scale))
    #if self.params.normalize_input_mean is not None:
    #  mean = K.variable(self.params.normalize_input_mean)
    #  self.enc_layers.insert(0, Lambda(lambda x:
    #       x - mean))
    #  self.dec_layers.append(Lambda(lambda x:
    #       x + mean))
    

    # input placeholder
    self.input = Input(shape=(self.params.input_dim,))

    # create encoder
    encoded = self.input
    for layer in self.enc_layers:
      encoded = layer(encoded)
    encoded_z_mean = self.enc_z_mean(encoded)
    encoded_z_log_var = self.enc_z_log_var(encoded)
    self.encoder = Model(input=self.input, output=encoded_z_mean)

    # create vae
    encoded_z_draw = self.enc_z_draw([encoded_z_mean, encoded_z_log_var])
    encoded_and_decoded = encoded_z_draw
    for layer in self.dec_layers:
      encoded_and_decoded = layer(encoded_and_decoded)
    encoded_and_decoded_x_mean = self.dec_x_mean(encoded_and_decoded)
    encoded_and_decoded_x_log_var = self.dec_x_log_var(encoded_and_decoded)
    
    kl_dist = StdGaussianKLDist(name='kl_dist') \
                               ([encoded_z_mean, encoded_z_log_var])
    log_prob = GaussianLogProb(name='log_prob') \
                              ([self.input, encoded_and_decoded_x_mean,
                                encoded_and_decoded_x_log_var])
    
    vae_loss = VAELoss(name='vae_loss')([kl_dist, log_prob])
    tot_loss = vae_loss
    if self.params.prediction_log_var_min is not None:
      x_log_var_regularizer = Lambda(lambda x: K.sum(K.relu(self.params.prediction_log_var_min-x),axis=-1))(encoded_and_decoded_x_log_var)
      tot_loss = SumLoss(name='tot_loss')([vae_loss, x_log_var_regularizer])
    self.vae_loss = Model(input=self.input, output=tot_loss)

    # create decoder
    self.latent_input = Input(shape=(self.params.latent_dim,))
    decoded = self.latent_input
    for layer in self.dec_layers:
      decoded = layer(decoded)
    decoded_x_mean = self.dec_x_mean(decoded)
    decoded_x_log_var = self.dec_x_log_var(decoded)
    self.decoder = Model(input=self.latent_input, output=decoded_x_mean)
    self.decoder_dist = Model(input=self.latent_input,
                              output=[decoded_x_mean, decoded_x_log_var])

    # create generator
    decoded_x_draw = self.dec_x_draw([decoded_x_mean, decoded_x_log_var])
    self.decoder_generator = Model(input=self.latent_input, output=decoded_x_draw)


  def learn(self, x,
            validation_split=0.0,
            validation_data=None,
            log_file_prefix=None,
            per_epoch_callback_funs=[],
            callbacks=[],
            verbose='print_epochs'  #one of: 'none'
                                    #        'print_epochs'
                                    #        'progress_bars'
            ):
    #reconstruction error
    #optimizer = optimizers.get({'class_name': str(self.params.optimizer),
    #                            'config': self.params.optimizer_params})
    optimizer = optimizers.get(self.params.optimizer,
                               self.params.optimizer_params)
    #self.vae_loss.compile(optimizer=SGD(lr=0.0001), loss=pass_through_loss)
    #self.vae_loss.compile(optimizer=Adam(lr=0.01), loss=pass_through_loss)
    #self.vae_loss.compile(optimizer=Adagrad(lr=0.01), loss=pass_through_loss)
    self.vae_loss.compile(optimizer=optimizer, loss=pass_through_loss)

    if validation_data is not None:
      validation_data = (validation_data, validation_data)
    
    validation = (validation_data is not None)

    # optionally add callbacks
    keras_callbacks = []
    # 'built-in' callbacks
    if log_file_prefix:
      #keras_callbacks.append(keras.callbacks.CSVLogger(log_file_prefix + ".log"))
      if self.params.log_weights:
        keras_callbacks.append(WeightLogger(self.vae_loss, log_file_prefix))
      if self.params.log_loss:
        keras_callbacks.append(LossLogger(log_file_prefix,
                                          per_patch=self.log_loss_per_patch))
      if self.params.log_loss and validation:
        keras_callbacks.append(LossLogger(log_file_prefix, loss='val_loss'))
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
    
    if self.reduce_lr_on_plateau:
      if self.reduce_lr_on_plateau == True:
        monitor = ('val_loss' if validation else 'loss')
      else:
        monitor = self.reduce_lr_on_plateau
      keras_callbacks.append(keras.callbacks.ReduceLROnPlateau(
          monitor=monitor,
          factor=self.reduce_lr_factor,
          patience=self.params.reduce_lr_patience))
    
    if verbose == 'none':
      verbose = 0
    elif verbose == 'print_epochs':
      verbose = 2
    elif verbose == 'progress_bars':
      verbose = 1
    else:
      assert False # invalid verbosity

    #train
    self.vae_loss.fit(x, x,
        nb_epoch=self.params.n_epochs,
        batch_size=self.params.batch_size,
        shuffle=True,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=keras_callbacks,
        verbose=verbose
        )

    #for l, layer in enumerate(self.vae_loss.layers):
    #  print("layer %s" % layer.name)
    #  for w in layer.get_weights():
    #    print(w)
    
  def encode(self, x):
    return self.encoder.predict(x)

  def decode(self, x):
    return self.decoder.predict(x)

  def decode_generate(self, x):
    return self.decoder_generator.predict(x)
  
  def save(self, filename):
    with open(filename + "_params.pkl", 'wb') as f:
      pickle.dump(self.params, f)
    self.vae_loss.save_weights(filename + "_weights.h5")

  def load(self, filename):
    with open(filename + "_params.pkl", 'rb') as f:
      self.params = pickle.load(f)
    # FIXME: REMOVE! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    self.params.prediction_log_var_min = np.log(0.1**2) 
    # FIXME: REMOVE! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    self.build_models()
    self.vae_loss.load_weights(filename + "_weights.h5")
    return self
