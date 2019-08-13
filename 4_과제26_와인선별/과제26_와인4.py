from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
currentPath = os.getcwd()
os.chdir('D:/workspace/Deep_Learning/과제26')

# 한글 사용하기
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')


# 데이터 입력
df_pre = pd.read_csv('dataset/wine.csv', header = None)
df_pre.head()

df = df_pre.sample(frac=1)
dataset = df.values
# 기존 wine.csv에서 11열(와인맛), 12열(class) 를 자리를 바꿔놓음
X = dataset[:,0:12]
Y = dataset[:,12]

from keras.utils import np_utils
Y = np_utils.to_categorical(Y,11)
Y[0]
# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 모델 설정
# hidden layer = 40,15 input_dim =12, 출력층 = 11)
model = Sequential()
model.add(Dense(35, input_dim = 12, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(11, activation='softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = './model4/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


# 모델 저장 조건
modelpath = "./model4/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=100)


# 모델 실행 및 저장
model.fit(X, Y,
          validation_split = 0.3,
          epochs = 2000,
          batch_size = 750,
          callbacks = [early_stopping_callback, checkpointer])


# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y)[1]))




### 그래프
history = model.fit(X, Y,
                    validation_split = 0.3,
                    epochs = 2000,
                    batch_size = 750)


y_vloss = history.history['val_loss']

y_acc = history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c = 'red', markersize = 3)
plt.plot(x_len, y_acc, 'o', c = 'blue', markersize = 3)
plt.xlabel('epochs')
plt.ylabel('정확도')
plt.title('와인_모델5')
plt.show()