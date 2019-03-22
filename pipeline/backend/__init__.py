
# import vanilla keras backend
from keras.backend import *

# import extra stuff
if backend() == 'theano':
  from .theano_backend import *
elif backend() == 'tensorflow':
  from .tensorflow_backend import *
else:
  raise Exception('Unknown backend: ' + str(_BACKEND))

