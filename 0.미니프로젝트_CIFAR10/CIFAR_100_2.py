# 라이브러리
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import regularizers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

currentPath = os.getcwd()
os.chdir('D:/workspace/Deep_Learning/0.미니프로젝트_CIFAR10')


# CIFAR10 데이터 로딩 및 확인


(X_train, y_train0), (X_test, y_test0) = cifar100.load_data()
print(X_train.shape, X_train.dtype)
print(y_train0.shape, y_train0.dtype)
print(X_test.shape, X_test.dtype)
print(y_test0.shape, y_test0.dtype)

# 데이터 확인 / interpolation = "bicubic" -> 인접한 16개 화소의 화소값과 거리에 따른 가중치의 곱을 사용
plt.subplot(141)
plt.imshow(X_train[0], interpolation="bicubic")
plt.grid(False)
plt.subplot(142)
plt.imshow(X_train[4], interpolation="bicubic")
plt.grid(False)
plt.subplot(143)
plt.imshow(X_train[8], interpolation="bicubic")
plt.grid(False)
plt.subplot(144)
plt.imshow(X_train[12], interpolation="bicubic")
plt.grid(False)
plt.show()


# 자료형 변환 및 스케일링

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

print(X_train.shape, X_train.dtype)

# one - hot - encoding
Y_train = np_utils.to_categorical(y_train0, 100)
Y_test = np_utils.to_categorical(y_test0, 100)
Y_train[:4]

# model 구현
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3),activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(Conv2D(64, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


hist = model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_test, Y_test), verbose=2)

print("Test ACCURACY : %.4f" % (model.evaluate(X_test, Y_test)[1]))
model.save("cifar10_2.hdf5")

# 테스트 셋의 오차
y_vloss = hist.history['val_loss']

# 학습셋의 오차
y_loss = hist.history['loss']

# 테스트 셋의 정확도
y_vacc = hist.history['val_acc']
# 학습셋의 정확도
y_acc = hist.history['acc']


# 학습셋 테스트셋 오차율 그래프

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, c='red', label  = 'Test set_loss')
plt.plot(x_len, y_loss, c='blue', label  = 'Train set_loss')

plt.title('Test, Train loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 학습셋 테스트셋 정확도 그래프
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vacc, c='red', label  = 'Test set_acc')
plt.plot(x_len, y_acc, c='blue', label  = 'Train set_acc')

plt.title('Test, Train Acc')
plt.legend(loc = 'low right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()



learn.show_results(rows=3,figsize=(12,10))
