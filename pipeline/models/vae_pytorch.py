'''
Variational auto-encoder.
'''

from types import SimpleNamespace
import math
import numpy as np
import pickle
import time, datetime
#import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


from common import auto_expand, print_err
#from .nncommon import WeightLogger, WeightDiffStatLogger, LossLogger, create_activation

def create_activation(name):
  if name.lower() == 'ReLU'.lower():
    return nn.ReLU()
  elif name.lower() == 'PReLU'.lower():
    return nn.PReLU()
  elif name.lower() == 'ELU'.lower():
    return nn.ELU()
  else:
    assert False

class Gaussian(nn.Module):
    def __init__(self, shape):
        super(Gaussian, self).__init__()
        self.shape = shape   
        self.cuda_ = False 
    
    def cpu():
      self.cuda_ = False
    
    def cuda(device_id=None):
      self.cuda_ = True
    
    def forward(self):
      if self.cuda_:
        return torch.cuda.FloatTensor(shape).normal_()
      else:
        return torch.FloatTensor(shape).normal_()

'''class Constant(nn.Module):
    def __init__(self, shape):
        super(Constant, self).__init__()
        self.shape = shape
        self.cuda_ = False 
    
    def cpu():
      self.cuda_ = False
    
    def cuda(device_id=None):
      self.cuda_ = True
    
    def forward(self):
      if self.cuda_:
        return torch.cuda.FloatTensor(shape).normal_()
      else:
        return torch.FloatTensor(shape).normal_()'''

def rand_normal_reparametrised(mean, log_var):
  std = log_var.mul(0.5).exp_()
  size = mean.size()
  if std.is_cuda:
    z0 = torch.cuda.FloatTensor(size).normal_()
  else:
    z0 = torch.FloatTensor(size).normal_()
  z0 = Variable(z0)
  return z0.mul(std).add_(mean)


def gaussian_log_prob(x, mean, log_var):
  return torch.sum(-(x - mean).pow(2) / (2 * log_var.exp()) -
                   0.5 * (math.log(2 * math.pi) + log_var), dim=1)

def std_gaussian_kl_dist(mean, log_var):
  return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

#recon_function = nn.MSELoss()

# loss function
def vae_loss(x, x_recon_mean, x_recon_log_var, z_mean, z_log_var):
  log_prob = gaussian_log_prob(x, x_recon_mean, x_recon_log_var)
  kl_dist = std_gaussian_kl_dist(z_mean, z_log_var)
  #kl_dist = 0 # FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
  #print("----")
  #print(torch.sum(log_prob).data[0])
  #print(torch.sum(kl_dist).data[0])
  return torch.sum(kl_dist - log_prob)
  #return recon_function(x_recon_mean, x)

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
           early_stopping_target='auto', #one of: 'auto'
                                         #        'loss'
                                         #        'val_loss'
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
    self.params.early_stopping_target = early_stopping_target
    self.reduce_lr_on_plateau = reduce_lr_on_plateau
    self.reduce_lr_factor = reduce_lr_factor
    self.reduce_lr_patience = reduce_lr_patience
    self.params.log_weights = log_weights
    self.params.log_loss = log_loss
    self.log_loss_per_patch = log_loss_per_patch

    self.build_models()

    return self


  def build_models(self):

    class VAEModel(nn.Module):
      def __init__(self, params):
        super(VAEModel, self).__init__()
        self.params = params

        # encoding layers
        self.enc_layers = nn.ModuleList()
        enc_dims = list(self.params.enc_dims)
        enc_activations = auto_expand(self.params.enc_activations)
        if self.params.input_dropout:
          self.enc_layers.append(nn.Dropout(self.params.input_dropout))
        last_dim = self.params.input_dim
        for i in range(self.params.n_enc):
          #add_bias_terms = not self.params.batch_normalization
          add_bias_terms = True
          self.enc_layers.append(nn.Linear(last_dim, enc_dims[i],
                                           bias=add_bias_terms))
          #'enc_hidden_%d'%(i+1)
          if i == 0:
            self.first_dense_enc_layer = self.enc_layers[-1]
          #if self.params.batch_normalization:
          #  self.enc_layers.append(BatchNormalization(mode=2))
          self.enc_layers.append(create_activation(enc_activations[i]))
          if self.params.enc_dropout:
            self.enc_layers.append(nn.Dropout(auto_expand(
                self.params.enc_dropout)[i]))
          last_dim = enc_dims[i]
        
        # latent layers
        self.enc_z_mean = nn.Linear(last_dim, self.params.latent_dim)
        self.enc_z_log_var = nn.Linear(last_dim, self.params.latent_dim)
        #self.enc_z_draw = Gaussian()
        
        # decoding layers
        self.dec_layers = nn.ModuleList()
        dec_dims = list(self.params.dec_dims)
        dec_activations = auto_expand(self.params.dec_activations)
        if self.params.latent_dropout:
          self.dec_layers.append(Dropout(self.params.latent_dropout))
        last_dim = self.params.latent_dim
        for i in range(self.params.n_dec):
          #add_bias_terms = not (self.params.batch_normalization and i != self.params.n_dec)
          add_bias_terms = True
          self.dec_layers.append(nn.Linear(last_dim, dec_dims[i],
                                           bias=add_bias_terms))
          if i == self.params.n_dec:
            self.last_dense_dec_layer = self.dec_layers[-1]
          #if self.params.batch_normalization and i != self.params.n_dec:
          #  self.dec_layers.append(BatchNormalization(mode=2))
          self.dec_layers.append(create_activation(dec_activations[i]))
          if self.params.dec_dropout:
            self.dec_layers.append(nn.Dropout(auto_expand(
                self.params.dec_dropout)[i]))
          last_dim = dec_dims[i]
        
        # prediction layers
        self.dec_x_mean = nn.Linear(last_dim, self.params.input_dim)
        #pvname = 'pred_log_var'
        if self.params.prediction_var in ['persample_independent', 'pi']:
          self.dec_x_log_var = nn.Linear(last_dim, self.params.input_dim)
          self.dec_x_log_var.weight.data.zero_()
        elif self.params.prediction_var in ['persample_same', 'ps']:
          self.dec_x_log_var = nn.Linear(last_dim, 1)
          self.dec_x_log_var.weight.data.zero_()
        elif self.params.prediction_var in ['global_independent', 'gi']:
          #self.dec_x_log_var = Const(shape=(self.params.input_dim,), trainable=True, name=pvname)
          self.dec_x_log_var = nn.Parameter(torch.Tensor(self.params.input_dim))
          self.dec_x_log_var.data.zero_()
        elif self.params.prediction_var in ['global_same', 'gs']:
          self.dec_x_log_var = nn.Parameter(torch.Tensor(1))
          self.dec_x_log_var.data.zero_()
        else:
          self.dec_x_log_var = Variable(torch.from_numpy(self.params.prediction_var).float(), requires_grad=False)
          #self.dec_x_log_var = torch.from_numpy(self.params.prediction_var).float()
        #self.dec_x_draw = Gaussian(name='pred_draw')

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
        self.cuda_ = False
      
      def cuda(self):
        self.cuda_ = True
        super(VAEModel, self).cuda()
      
      def cpu(self):
        self.cuda_ = False
        super(VAEModel, self).cpu()

      def encode(self, x):
        encoded = x
        for layer in self.enc_layers:
          encoded = layer(encoded)
        encoded_z_mean = self.enc_z_mean(encoded)
        encoded_z_log_var = self.enc_z_log_var(encoded)
        return encoded_z_mean, encoded_z_log_var

      def decode(self, z):
        decoded = z
        for layer in self.dec_layers:
          decoded = layer(decoded)
        decoded_x_mean = self.dec_x_mean(decoded)
        if isinstance(self.dec_x_log_var, nn.Module):
          decoded_x_log_var = self.dec_x_log_var(decoded)
        else:
          decoded_x_log_var = self.dec_x_log_var
        return decoded_x_mean, decoded_x_log_var

      def forward(self, x):
        #encoded_z_mean, encoded_z_log_var = self.encode(x.view(-1, self.params.input_dim))
        encoded_z_mean, encoded_z_log_var = self.encode(x)
        z = rand_normal_reparametrised(encoded_z_mean, encoded_z_log_var)
        return self.decode(z) + (encoded_z_mean, encoded_z_log_var)

    
    self.model = VAEModel(self.params)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      print_err("GPU available -> using CUDA", flush=True)
      self.cuda = True
      self.model.cuda()
    else:
      self.cuda = False
      print_err("no GPU available -> using CPU", flush=True)
      

    '''kl_dist = StdGaussianKLDist(name='kl_dist') \
    
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

    # create generator
    decoded_x_draw = self.dec_x_draw([decoded_x_mean, decoded_x_log_var])
    self.decoder_generator = Model(input=self.latent_input, output=decoded_x_draw)'''

  def use_gpu(self):
    if self.cuda == False:
      self.cuda = True
      self.model.cuda()
  
  def use_cpu(self):
    if self.cuda == True:
      self.cuda = False
      self.model.cpu()

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
    optim_fun = getattr(optim, self.params.optimizer)
    optimizer = optim_fun(self.model.parameters(),
                          **self.params.optimizer_params)

    if validation_split > 0:
      assert validation_data is None
      perm = np.random.permutation(x.shape[0])
      split_idx = round(validation_split * x.shape[0])
      val, train = perm[:split_idx], perm[split_idx:]
      validation_data, x = x[val,:], x[train,:]

    validation = (validation_data is not None)

    # optionally add callbacks
    keras_callbacks = []
    # 'built-in' callbacks
    '''if log_file_prefix:
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
          patience=self.params.reduce_lr_patience))'''
    
    '''if verbose == 'none':
      verbose = 0
    elif verbose == 'print_epochs':
      verbose = 2
    elif verbose == 'progress_bars':
      verbose = 1
    else:
      assert False # invalid verbosity'''

    loss_fun = vae_loss
    if self.params.prediction_log_var_min is not None:
      # TODO: Now reqularization strength depends on the type (size) of
      # x_log_var. Should probably unify?
      x_log_var_regularizer = lambda xlv: (
                                torch.sum(F.relu(
                                  self.params.prediction_log_var_min-xlv
                                )))
      loss_fun = lambda x, x_mean, x_log_var, z_mean, z_log_var: (
                   vae_loss(x, x_mean, x_log_var, z_mean, z_log_var) +
                   x_log_var_regularizer(x_log_var))

    if self.cuda:
      import torch.backends.cudnn as cudnn
      #cudnn.benchmark = True

    #train
    progress_interval = 100
    n_samples = x.shape[0]
    n_patches = n_samples // self.params.batch_size
    best_es_target_loss = float('inf')
    epochs_no_improvement = 0
    #print(self.model)
    start_time = time.perf_counter()
    for epoch in range(self.params.n_epochs):
      #print(self.model.state_dict())
      # train
      epoch_start_time = time.perf_counter()
      self.model.train()
      train_loss = 0
      perm = np.random.permutation(x.shape[0])
      batches = [perm[i:i + self.params.batch_size]
                 for i in range(0, len(perm), self.params.batch_size)]
      for batch_idx, batch_rows in enumerate(batches):
        batch_data = torch.from_numpy(x[batch_rows, :]).float()
        batch_data = Variable(batch_data)
        if self.cuda:
          batch_data = batch_data.cuda()
        optimizer.zero_grad()
        recon_mean, recon_log_var, z_mean, z_log_var = self.model(batch_data)
        loss = loss_fun(batch_data, recon_mean, recon_log_var, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if (batch_idx+1) % progress_interval == 0:
          #print(loss.data[0])
          print_err('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * self.params.batch_size, n_samples,
              100. * batch_idx / len(batches),
              loss.data[0] / len(batch_rows)), flush=True)

      train_loss /= n_samples
      print_err("Epoch: %4d  Average loss: %.3f" % (
                epoch, train_loss), end='', flush=True)
      
      
      # validation
      if validation:
        self.model.eval()
        val_loss = 0
        perm = range(validation_data.shape[0])
        batches = [perm[i:i + self.params.batch_size]
                  for i in range(0, len(perm), self.params.batch_size)]
        for batch_idx, batch_rows in enumerate(batches):
          batch_data = torch.from_numpy(validation_data[batch_rows, :]).float()
          batch_data = Variable(batch_data, volatile=True)
          if self.cuda:
            batch_data = batch_data.cuda()
          recon_mean, recon_log_var, z_mean, z_log_var = self.model(batch_data)
          loss = loss_fun(batch_data, recon_mean, recon_log_var, z_mean, z_log_var)
          val_loss += loss.data[0]
        
        val_loss /= validation_data.shape[0]
        print_err("  Validation loss: %.4f" % (val_loss),
                  end='', flush=True)
      

      epoch_elapsed = time.perf_counter() - epoch_start_time
      elapsed_str = str(datetime.timedelta(seconds=epoch_elapsed))
      print_err("  Time: %s" % (elapsed_str), flush=True)

      if self.params.early_stopping:
        if self.params.early_stopping_target == 'auto':
          if validation:
            es_target_loss = val_loss
          else:
            es_target_loss = loss
        elif self.params.early_stopping_target == 'val_loss':
          es_target_loss = val_loss
        elif self.params.early_stopping_target == 'loss':
          es_target_loss = train_loss
        else:
          assert False, "invalid early_stopping_target"

        if es_target_loss < best_es_target_loss:
          best_es_target_loss = es_target_loss
          best_state = self.model.state_dict()
          epochs_no_improvement = 0
        else:
          epochs_no_improvement += 1
          if epochs_no_improvement > self.params.early_stopping_patience:
            self.model.load_state_dict(best_state)
            print("Early stopping as no improvement in %d epochs" % 
                  (self.params.early_stopping_patience))
            break
    
    #print(self.model.state_dict())

    '''self.vae_loss.fit(x, x,
        nb_epoch=self.params.n_epochs,
        batch_size=self.params.batch_size,
        shuffle=True,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=keras_callbacks,
        verbose=verbose
        )'''

    #for l, layer in enumerate(self.vae_loss.layers):
    #  print("layer %s" % layer.name)
    #  for w in layer.get_weights():
    #    print(w)
    
  def encode(self, x):
    x = Variable(torch.from_numpy(x).float(), volatile=True)
    if self.cuda:
      x = x.cuda()
    z_mean, z_log_var = self.model.encode(x)
    return z_mean.data.cpu().numpy()

  def encode_generate(self, x):
    x = Variable(torch.from_numpy(x).float(), volatile=True)
    if self.cuda:
      x = x.cuda()
    z_mean, z_log_var = self.model.encode(x)
    z = rand_normal_reparametrised(z_mean, z_log_var)
    return z.data.cpu().numpy()

  def decode(self, z):
    z = Variable(torch.from_numpy(z).float(), volatile=True)
    if self.cuda:
      z = z.cuda()
    x_mean, x_log_var = self.model.decode(z)
    return x_mean.data.cpu().numpy()

  def decode_generate(self, z):
    z = Variable(torch.from_numpy(z).float(), volatile=True)
    if self.cuda:
      z = z.cuda()
    x_mean, x_log_var = self.model.decode(z)
    x = rand_normal_reparametrised(x_mean, x_log_var)
    return x.data.cpu().numpy()
  
  def save(self, filename):
    with open(filename + "_params.pkl", 'wb') as f:
      pickle.dump(self.params, f)
    #with open(filename + "_weights.pkl", 'wb') as f:
    #  pickle.dump(self.model.state_dict(), f)
    torch.save(self.model.state_dict(), filename + "_weights.pkl")

  def load(self, filename):
    with open(filename + "_params.pkl", 'rb') as f:
      self.params = pickle.load(f)
    self.build_models()
    #with open(filename + "_weights.pkl", 'rb') as f:
    #  self.model.load_state_dict(pickle.load(f))
    self.model.load_state_dict(torch.load(filename + "_weights.pkl"))
    return self
