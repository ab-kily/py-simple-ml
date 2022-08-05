# Own implementation of  LogisticRegression
import time
import numpy as np

class LogisticRegression:
  def __init__(self,epoch=1000,learning_rate=0.01,learn_method='sgd',stop_rate=0.0001):
    self.epoch = epoch # number of learning steps
    self.learn_method = learn_method # keep learn method here
    self.learning_rate = learning_rate # learning rate
    self.stop_rate = stop_rate # rate to stop learning
    self.learn_func = None # reference to learning func
    self.cost_func = None # fererence to cost func

    self.weights = None
    self.epoch_passed = 0
    self.learn_time = 0

    if(self.learn_method == 'sgd'):
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
    set initial weights
  '''
  def init_weights(self,X):
    #return np.random.randn(X.shape[1], 1);
    return np.zeros((X.shape[1], 1))

  '''
    learn function
  '''
  def fit(self,X,Y):
    X = self.addones(X)
    Y = np.reshape(Y, (len(Y), 1))
    W = self.init_weights(X)

    self.epoch_passed = 0
    self.learn_time = 0

    start = time.perf_counter()
    self.learn_func(X,Y,W)
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
    X = self.addones(X)

    return self.sigmoid_linear_regression(X,self.weights)

  '''
    predicts result baased on input
  '''
  def predict(self,X):
    probas = self.predict_proba(X)
    for idx, y in enumerate(probas):
      probas[idx] = 1 if y > 0.5 else 0

    return probas

  '''
    adds one to features matrix
  '''
  def addones(self,X):
    return np.hstack((np.ones((X.shape[0],1)),X))

  '''
    SGD implementation
  '''
  def learn_sgd(self,X,Y,W):
    prev = self.cost_binary_cross_entropy(X,Y,W)
    for enum in range(self.epoch):
      self.epoch_passed = enum+1
      gradients = self.gradient(X,Y,W)
      W = W-self.learning_rate*gradients
      actual = self.cost_binary_cross_entropy(X,Y,W)
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
    binary cross-entropy
  '''
  def cost_binary_cross_entropy(self,X,Y,W):
    m = X.shape[0]
    total_cost = -(1 / m) * np.sum(
        Y * np.log(self.sigmoid_linear_regression(X, W)) + (1 - Y) * np.log(
            1 - self.sigmoid_linear_regression(X, W)))
    return total_cost

  '''
    derevative of binary cross entropy
  '''
  def gradient(self, X,Y,W):
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, self.sigmoid_linear_regression(X,W) - Y)

  '''
    linear regression passed through sigmoid
  '''
  def sigmoid_linear_regression(self,X,W):
    return self.sigmoid(self.linear_regression(X,W))

  '''
    linear regression
  '''
  def linear_regression(self,X,W):
    return np.dot(X,W)

  '''
    sigmoid
  '''
  def sigmoid(self,H):
    return 1/(1+np.exp(-H))

