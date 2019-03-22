import numpy as np
import matplotlib.pyplot as plt

x = np.load("param_opt/opt_params.npy")
y = np.load("param_opt/opt_results.npy")

domain = [
  {'name': 'learning_rate_log10', 'type': 'continuous', 'domain': (-5,-1)},
  {'name': 'n_hidden_layers', 'type': 'discrete', 'domain': [1, 2, 3]},
#  {'name': 'repr_dim', 'type': 'continuous', 'domain': (1, 10)},
  {'name': 'repr_dim', 'type': 'discrete', 'domain': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]},
  {'name': 'hidden_layer_size_mul_log10', 'type': 'continuous', 'domain': (0,4)},
  ]

assert x.shape[1] == len(domain)

(fig, ax) = plt.subplots(1, len(domain), sharey=True)

for i in range(len(domain)):
  ax[i].scatter(x[:,i], y)
  ax[i].set_xlabel(domain[i]['name'])

plt.show()

