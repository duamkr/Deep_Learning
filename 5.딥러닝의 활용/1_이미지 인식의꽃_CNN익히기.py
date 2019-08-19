

##### mnist #####

from keras.datasets import mnist

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

X_train.shape, X_test.shape          # 28 x 28 픽셀 크기

# X_train 의 첫번쨰 그림
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap = 'Greys')
plt.show()

# X_train 의 구성 0~255로 표시됨
for x in X_train[0] :
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')


# X_train 28 x 28 을 1차원으로 바꿔줌
X_train = X_train.reshape(X_train.shape[0], 784)
# X_train 데이터 분산화 (0~255의 이루어진 값을 0~1사이의 값으로 바꾸는 작업을 정규화라고 함)
X_train = X_train.astype('float64')
X_train = X_train / 255
# X_test도 마찬가지로 작업
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255



# Y_train, Y_test one-hot-encoding
from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_class_train,10)
Y_test = np_utils.to_categorical(Y_class_test,10)




# 모델 실행
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255


Y_train = np_utils.to_categorical(Y_class_train,10)
Y_test = np_utils.to_categorical(Y_class_test,10)


model = Sequential()
model.add(Dense(512, input_dim = 784, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

import os
from keras.callbacks import ModelCheckpoint,EarlyStopping

MODEl_DIR = './model/'
if not os.path.exists((MODEl_DIR)) :
    os.mkdir(MODEl_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only=True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)

history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 0,
                    callbacks = [early_stopping_callback,checkpointer])

print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))

import matplotlib.pyplot as plt
import numpy as np

y_vloss = history.history['val_loss']

y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = "red", label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = "blue", label = 'Trainset_loss')

plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()