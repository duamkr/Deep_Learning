# 폐암 수술 환자의 생존율 예측하기 (p.29)

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow as tf
if type(tf.contrib) != type(tf):
    tf.contrib_warning = None



seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

Data_set = np.loadtxt('data/ThoraricSurgery.csv', delimiter = ',')
Data_set

# 환자의 기록과 수술 결과물을 X,Y를 구분 저장
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정합니다.(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim = 17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝을 실행합니다.
model.compile(loss = 'mean_squared_error', optimizer = 'adam',
metrics = ['accuracy'])
model.fit(X,Y, epochs = 30, batch_size=10)

# 결과를 출력합니다
print('\n Accuracy: %.4f' % (model.evaluate(X,Y)[1]))