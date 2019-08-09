
import pandas as pd
df = pd.read_csv('data/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue = 'species')


dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]


# setosa, versicolor, virginaca로 되어있는 Y를 0,1,2의 숫자로 변경해줌
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 활성화 함수는 0,1로 이루어져 있어야 하므로 0,1,2로 변경해준 Y를 one - hot - encoding
# one - hot - encoding 으로 0,1,2 였던 Y를 array[0,0,1][0,1,0][0,0,1] 변경
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y)


# 소프트맥스

model = Sequential()
model.add(Dense(16, input_dim=4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))


# 아이리스 품종 예측 전체
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('data/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.head()

sns.pairplot(df, hue = 'species')


dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)


model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X, Y_encoded, epochs = 50, batch_size = 1)

print("\n Accuracy : %.4f" % (model.evaluate(X, Y_encoded)[1]))