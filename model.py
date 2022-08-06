from sklearn import datasets # for iris dataset
from sklearn.preprocessing import LabelEncoder # for label encoding
from sklearn.model_selection import train_test_split # to split dataset
import numpy as np
import pandas

from mylearn import LogisticRegression

# loading iris dataset
ds = datasets.load_iris()

# convert to pandas DataFrame to make it easy to work with
names = list(map(lambda t : [ds['target_names'][int(t)]],ds['target']))
data = pandas.DataFrame(data=np.hstack((ds['data'],names)),
                       columns=ds['feature_names'] + ['target'])

# convert columns back to float
for feature in ds['feature_names']:
  data[feature] = data[feature].astype(float)

# keep only 2 classes in this dataset "Iris Versicolor" and "Iris Virginica"
data = data[data['target'] != 'setosa'].reset_index(drop=True)

# dropping NaN values
# data['sepal length (cm)'].unique()
data = data.dropna()

# these are features (X)
features_x = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# this is result (Y)
result_y = 'target'

# get all columns except target one
BASE_X = data[features_x]

# convert target using LabelEncoder
le = LabelEncoder()
le.fit(data[result_y])
BASE_Y = le.transform(data[result_y])

# splitting dataset train 80% test 20%
TRAIN_X, TEST_X, TRAIN_Y, TEST_Y = train_test_split(BASE_X, BASE_Y, test_size = 0.2, random_state = 0)

# preparing table to store results
results = pandas.DataFrame(columns=['Method','Score','Iterations','Time'])

# Learning using Gradient Descent (GD)
my_model = LogisticRegression(learn_method='gd',epoch=2000)
my_model.fit(TRAIN_X,TRAIN_Y)
my_score = my_model.score(TEST_X,TEST_Y)
my_iter = my_model.n_iter()
my_time = my_model.time()
print("SGD score is: {}".format(my_score))
print("SGD iter number: {}".format(my_iter))
print("SGD time: {} sec".format(my_time))
results.loc[len(results)] = ['MY GD',my_score,my_iter,my_time]

# Learning using Stochastic Gradient Descent (SGD)
my_model = LogisticRegression(learn_method='sgd',epoch=2000)
my_model.fit(TRAIN_X,TRAIN_Y)
my_score = my_model.score(TEST_X,TEST_Y)
my_iter = my_model.n_iter()
my_time = my_model.time()
print("SGD score is: {}".format(my_score))
print("SGD iter number: {}".format(my_iter))
print("SGD time: {} sec".format(my_time))
results.loc[len(results)] = ['MY SGD',my_score,my_iter,my_time]

# RMSPROP
my_model = LogisticRegression(learn_method='rmsprop',epoch=2000)
my_model.fit(TRAIN_X,TRAIN_Y)
my_score = my_model.score(TEST_X,TEST_Y)
my_iter = my_model.n_iter()
my_time = my_model.time()
print("RMSPROP score is: {}".format(my_score))
print("RMSPROP iter number: {}".format(my_iter))
print("RMSPROP time: {} sec".format(my_time))
results.loc[len(results)] = ['MY RMSPROP',my_score,my_iter,my_time]

# ADAM
my_model = LogisticRegression(learn_method='adam',epoch=2000)
my_model.fit(TRAIN_X,TRAIN_Y)
my_score = my_model.score(TEST_X,TEST_Y)
my_iter = my_model.n_iter()
my_time = my_model.time()
print("ADAM score is: {}".format(my_score))
print("ADAM iter number: {}".format(my_iter))
print("ADAM time: {} sec".format(my_time))
results.loc[len(results)] = ['MY ADAM',my_score,my_iter,my_time]

# NADAM
my_model = LogisticRegression(learn_method='nadam',epoch=2000)
my_model.fit(TRAIN_X,TRAIN_Y)
my_score = my_model.score(TEST_X,TEST_Y)
my_iter = my_model.n_iter()
my_time = my_model.time()
print("NADAM score is: {}".format(my_score))
print("NADAM iter number: {}".format(my_iter))
print("NADAM time: {} sec".format(my_time))
results.loc[len(results)] = ['MY NADAM',my_score,my_iter,my_time]

print(results)
