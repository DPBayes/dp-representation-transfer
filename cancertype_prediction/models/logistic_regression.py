import numpy as np
import pickle
from scipy.optimize import fmin_bfgs

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

class DPLogisticRegression:
  def init(self, input_dim, classes, alpha, epsilon):
    #self.params = SimepleNamespace()
    self.input_dim = input_dim
    assert len(classes) == 2
    self.classes = np.array(classes)
    self.alpha = alpha
    self.epsilon = epsilon
    self.betas = np.full((input_dim+1,), np.nan)
    return self
  
  def fit(self, x, y):
    self.learn(x, y)
    return self

  def learn(self, x, y,
            validation_split=0.0, # unused
            validation_data=None, # unused
            log_file_prefix=None, # unused
            deadline=None, # unused
            max_duration=None): # unused
    assert validation_split == 0.0

    d = x.shape[1]
    assert d == self.input_dim
    n_samples = x.shape[0]

    assert np.all(np.sum(x ** 2, axis=1) <= 1)#, ("max norm = %g" %
                                              #   (np.amax(np.sum(x ** 2, axis=1))))

    assert all(np.isin(y, self.classes))
    y = (y == self.classes[1]) * 2 - 1     # y is -1 or 1

    # add a dimension for bias term and renormalize to 1-ball
    x = np.hstack((x, np.ones((x.shape[0],1)))) / np.sqrt(2)
    assert np.all(np.sum(x ** 2, axis=1) <= 1.001)

    #print(y)
    #print(x)

    # randomize perturbation
    #temp_norm = np.random.gamma(d, 2 / self.epsilon, 1)
    temp_norm = np.random.gamma(d + 1, 2 / self.epsilon, 1)
    temp_angle = np.random.randn(d + 1)
    #temp_angle[d] = 0
    self.kappa = temp_norm * temp_angle / np.sqrt(sum(temp_angle ** 2))
    #print(self.kappa)

    # computes perturbed score
    def f(betas):
      # likelihood
      #l = np.sum(np.log(sigmoid(y * (np.dot(x, betas[:-1]) + betas[-1]))))
      l = np.sum(np.log(sigmoid(y * np.dot(x, betas))))
      # regularizer + perturbation

      # regularizer (bias term not regularized)
      l -= self.alpha / 2.0 * np.sum(betas[:-1] ** 2)
      # perturbation
      l -= np.dot(betas, self.kappa)
      return -l
    
    # computes the gradient of perturbed score
    def f_grad(betas):
      return ((np.arange(d+1) < d) * self.alpha * betas + self.kappa -
               np.dot(y * sigmoid(-y * np.dot(x, betas)), x))
 
    # Optimize
    betas_init = np.zeros(self.input_dim + 1)
    self.betas = fmin_bfgs(f, betas_init, fprime=f_grad, disp=False)
    #self.betas = fmin_bfgs(f, betas_init, disp=False)
    #print(self.betas)

  def predict(self, x):
    prob = sigmoid(np.dot(x, self.betas[:-1]) + self.betas[-1])
    #print(prob)
    return self.classes[np.round(prob).astype(int)]

  def predict_proba(self, x):
    assert False, "not implemented"
  
  def score(self, x, y, sample_weight=None):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, self.predict(x), sample_weight=sample_weight)
  
  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.input_dim, f)
      pickle.dump(self.alpha, f)
      pickle.dump(self.betas, f)
  
  def load (self, filename):
    with open(filename, 'rb') as f:
      self.input_dim = pickle.load(f)
      self.alpha = pickle.load(f)
      self.betas = pickle.load(f)
    return self

