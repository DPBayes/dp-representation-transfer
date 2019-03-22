'''
Simple 1-layer autoencoder for cluster. Set parameters in the master files.
'''

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import keras.callbacks
from keras import regularizers
import numpy as np

#from keras import backend as K
import backend as K
from keras.engine.topology import Layer

class Scale(Layer):
  def __init__(self, **kwargs):
    super(Scale, self).__init__(**kwargs)

  def build(self, input_shape):
    input_dim = input_shape[1]
    initial_weight_value = np.random.random((input_dim))
    self.W = K.variable(initial_weight_value)
    self.trainable_weights = [self.W]

  def call(self, x, mask=None):
    return K.mul(x, self.W)

  def get_output_shape_for(self, input_shape):
    return input_shape


class OneLayerAE:
  def init(self, input_dim, output_dim, enc_activation, dec_activation, n_epochs, batch_size):
    #input placeholder
    input = Input(shape=(input_dim,))

    #basic encoded representation of the input
    encoded = Dense(output_dim, activation=enc_activation)(input)

    #reconstructions
    decoder_layer = Dense(input_dim, activation=dec_activation)
    encoded_and_decoded = decoder_layer(encoded)
    #decoder_layer2 = Dense(input_dim, activation='tanh')
    #decoder_layer1 = Scale()
    #encoded_and_decoded = decoder_layer1(decoder_layer2(encoded))

    #model that maps input -> reconstruction
    self.autoencoder = Model(input=input, output=encoded_and_decoded)

    #separate encoder model
    self.encoder = Model(input=input, output=encoded)

    #placeholder for enc. input
    encoded_input = Input(shape=(output_dim,))

    # decoding only
    decoded = decoder_layer(encoded_input)
    #decoded = decoder_layer1(decoder_layer2(encoded_input))

    #separate decoder model
    self.decoder = Model(input=encoded_input, output=decoded)

    self.n_epochs = n_epochs
    self.batch_size = batch_size

    return self

  def learn(self, x, callbacks=[]):  
    #reconstruction error
    self.autoencoder.compile(optimizer='adadelta',loss='mean_squared_error')

    keras_callbacks = []
    for callback in callbacks:
      class CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
          callback()
      keras_callbacks.append(CB())
      callback()

    #train
    self.autoencoder.fit(x, x,
        nb_epoch=self.n_epochs,
        batch_size=self.batch_size,
        shuffle=True,
        callbacks=keras_callbacks
        )
    
    #for layer in self.autoencoder.layers:
    #  print(layer.get_weights())
    
  def encode(self, x):
      return self.encoder.predict(x)

  def decode(self, x):
      return self.decoder.predict(x)
  
  def save(self, filename):
    self.encoder.save_weights(filename)

  def load(self, filename):
    self.encoder.load_weights(filename)
    return self
