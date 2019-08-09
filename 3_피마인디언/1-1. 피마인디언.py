import pandas as pd
import numpy as np
import tensorflow as tf
if type(tf.contrib) != type(tf):
    tf.contrib_warning = None

df = pd.read_csv('data/pima-indians-diabetes.csv',
                 names = ["pregnant", "plasma", "pressure", "thickness",
                          "Insulin", "BMI", "pedigree", "age", "class"])

# 데이터 확인
df.head(5)
df.describe()
df[['pregnant', 'class']]

# 데이터 가공하기
# 임신횟수와 당뇨병 발병확률
df[["pregnant","class"]].groupby(["pregnant"], as_index = False).mean().sort_values(by='pregnant', ascending = True)


# matplotlib를 이용해 그래프 표현하기

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize = (12,12))
sns.heatmap(df.corr(), linewidths = 0.1, vmax = 0.5, cmap = plt.cm.gist_heat, linecolor = 'white', annot=True)\
plt.show()

# 공복혈당(plasma)과 당뇨병과의 관계
gird = sns.FacetGrid(df, col= 'class')
gird.map(plt.hist, 'plasma', bins =10)
plt.show()


# 피마 인디언의 당뇨병 예측 실행
# seed값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt('data/pima-indians-diabetes.csv',
                        delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y, epochs = 200, batch_size = 10)


# 결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(X, Y)[1]))