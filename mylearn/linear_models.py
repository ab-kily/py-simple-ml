# Own implementation of  LogisticRegression
import time
import numpy as np

from .tools import addones
from .gd_implementation import learn_base_gd
from .linear_implementation import linear_regression_matrix
from .linear_implementation import linear_regression_iterable
from .loss_implementation import binary_cross_entropy_matrix
from .loss_implementation import binary_cross_entropy_derivative_matrix
from .loss_implementation import binary_cross_entropy_iterable
from .loss_implementation import binary_cross_entropy_derivative_iterable

class LogisticRegression:
  def __init__(self,epoch=1000,learning_rate=0.01,learn_method='base_gd_matrix',stop_rate=0.0001):
    self.epoch = epoch # number of learning steps
    self.learn_method = learn_method # keep learn method here
    self.learning_rate = learning_rate # learning rate
    self.stop_rate = stop_rate # rate to stop learning
    self.learn_func = None # reference to learning func
    self.linear_func = None # reference to linear func
    self.loss_func = None # reference to loss function
    self.derivative_func = None # reference to derivative of loss function

    self.weights = None
    self.epoch_passed = 0
    self.learn_time = 0

    if(self.learn_method == 'base_gd_matrix'):
      self.learn_func = learn_base_gd
      self.linear_func = lambda X,W: self.sigmoid(linear_regression_matrix(X,W))
      self.loss_func = binary_cross_entropy_matrix
      self.derivative_func = binary_cross_entropy_derivative_matrix
    elif(self.learn_method == 'base_gd_iterable'):
      self.learn_func = learn_base_gd
      self.linear_func = lambda X,W: self.sigmoid(linear_regression_iterable(X,W))
      self.loss_func = binary_cross_entropy_iterable
      self.derivative_func = binary_cross_entropy_derivative_iterable
    elif(self.learn_method == 'sgd'):
      self.learn_func = self.learn_sgd
    elif(self.learn_method == 'rmsprop'):
      self.learn_func = self.learn_rmsprop
    elif(self.learn_method == 'adam'):
      self.learn_func = self.learn_adam
    elif(self.learn_method == 'nadam'):
      self.learn_func = self.learn_nadam
    else:
      raise Exception('Unknown learining method: {}'.format(self.learn_method))

  '''
    learn function
  '''
  def fit(self,X,Y):
    self.epoch_passed = 0
    self.learn_time = 0

    start = time.perf_counter()
    self.weights, self.epoch_passed = self.learn_func(
        X=X,Y=Y,
        n_epoch=self.epoch,
        learn_rate=self.learning_rate,
        stop_rate=self.stop_rate,
        linear_func=self.linear_func,
        loss_func=self.loss_func,
        derivative_func=self.derivative_func)
    self.learn_time = time.perf_counter() - start


  '''
    returns prediction score
  '''
  def score(self,X,Y):
    predictions = self.predict(X)
    scores = []
    for idx, y_pred in enumerate(predictions):
      # заполняем True если предсказание верное, иначе False
      scores.append(True if y_pred == Y[idx] else False)

    return scores.count(True)/len(scores)

  '''
    returns count of iterations passed
  '''
  def n_iter(self):
    return self.epoch_passed

  '''
    returns learn time
  '''
  def time(self):
    return self.learn_time

  '''
    returns prediction probabilities
  '''
  def predict_proba(self,X):
    return self.linear_func(X,self.weights)

  '''
    predicts result baased on input
  '''
  def predict(self,X):
    probas = self.predict_proba(X)
    for idx, y in enumerate(probas):
      probas[idx] = 1 if y > 0.5 else 0

    return probas


  '''
    SGD implementation
  '''
  def learn_sgd(self,X,Y,W):
    n = len(Y)
    indices = np.random.choice(range(0,n-1),size=int(0.1*n))
    X_SUBSET = np.array([X[i] for i in indices])
    Y_SUBSET = np.array([Y[i] for i in indices])

    prev = self.cost_binary_cross_entropy(X_SUBSET,Y_SUBSET,W)
    for enum in range(self.epoch):
      self.epoch_passed = enum+1
      gradients = self.gradient(X_SUBSET,Y_SUBSET,W)
      W = W-self.learning_rate*gradients
      actual = self.cost_binary_cross_entropy(X_SUBSET,Y_SUBSET,W)
      diff = abs(prev - actual)
      if(diff <= self.stop_rate):
        break
      prev = actual

    self.weights = W
    return W

  '''
    RMSPROP implementation
  '''
  def learn_rmsprop(self,X,Y,W):
    prev = self.cost_binary_cross_entropy(X,Y,W)
    lr = self.learning_rate
    cached_rmsprop = [0] * len(W)
    decay_rate = 0.9
    for enum in range(self.epoch):
      self.epoch_passed = enum+1

      gradients = self.gradient(X,Y,W)
      NEW_W = []
      for i, (w, grad) in enumerate(zip(W, gradients)):
        cached_rmsprop[i] = decay_rate * cached_rmsprop[i] + (1-decay_rate) * grad **2
        new_w = w-lr*grad/(np.sqrt(cached_rmsprop[i]+1e-6))
        NEW_W.append(new_w)

      W = NEW_W
      actual = self.cost_binary_cross_entropy(X,Y,W)
      diff = abs(prev - actual)
      if(diff <= self.stop_rate):
        break
      prev = actual

    self.weights = W
    return W

  '''
    ADAM implementation
  '''
  def learn_adam(self,X,Y,W):
    prev = self.cost_binary_cross_entropy(X,Y,W)
    lr = self.learning_rate
    m = [0] * len(W)
    v = [0] * len(W)
    t = 1
    beta1 = 0.9
    beta2 = 0.999
    for enum in range(self.epoch):
      self.epoch_passed = enum+1

      gradients = self.gradient(X,Y,W)
      NEW_W = []
      for i, (w, grad) in enumerate(zip(W, gradients)):
        m[i] = beta1 * m[i] + (1-beta1) * grad
        v[i] = beta2 * v[i] + (1-beta2) * grad **2
        m_corrected = m[i]/(1-beta1**t)
        v_corrected = v[i]/(1-beta2**t)

        new_w = w-lr*m_corrected/(np.sqrt(v_corrected+1e-8))

        NEW_W.append(new_w)

      t += 1

      W = NEW_W
      actual = self.cost_binary_cross_entropy(X,Y,W)
      diff = abs(prev - actual)
      if(diff <= self.stop_rate):
        break
      prev = actual

    self.weights = W
    return W

  '''
    NADAM implementation
  '''
  def learn_nadam(self,X,Y,W):
    prev = self.cost_binary_cross_entropy(X,Y,W)
    lr = self.learning_rate
    m = [0] * len(W)
    v = [0] * len(W)
    t = 1
    beta1 = 0.9
    beta2 = 0.999
    for enum in range(self.epoch):
      self.epoch_passed = enum+1

      gradients = self.gradient(X,Y,W)
      NEW_W = []
      for i, (w, grad) in enumerate(zip(W, gradients)):
        m[i] = beta1 * m[i] + (1-beta1) * grad
        v[i] = beta2 * v[i] + (1-beta2) * grad **2
        m_corrected = m[i]/(1-beta1**t)
        v_corrected = v[i]/(1-beta2**t)

        new_w = w - (lr/(np.sqrt(v_corrected)+1e-8)*(beta1*m_corrected+(1-beta1)*grad/(1-beta1**t)))

        NEW_W.append(new_w)

      t += 1

      W = NEW_W
      actual = self.cost_binary_cross_entropy(X,Y,W)
      diff = abs(prev - actual)
      if(diff <= self.stop_rate):
        break
      prev = actual

    self.weights = W
    return W

  '''
    linear regression passed through sigmoid
  '''
  def sigmoid_linear_regression(self,X,W):
    return self.sigmoid(linear_regression_iterable(X,W))

  '''
    sigmoid
  '''
  def sigmoid(self,H):
    return 1/(1+np.exp(-H))

