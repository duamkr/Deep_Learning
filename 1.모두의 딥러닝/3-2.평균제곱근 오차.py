

############ 평균 제곱근 오차 ############

import numpy as np

# 기울기 a와 절편 b

ab = [3 ,76]

# x,y의 데이터 값
data = [[2,81], [4,93], [6, 91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y = ax + b 에 a와 b값을 대입하여 결과를 출력하는 함수
def predict(x) :
    return ab[0] * x + ab[1]


# RMSE 함수

def rmse(p,a) :
    return np.sqrt(((p - a)** 2).mean())

# RMSE함수를 각 y값에 대입하여 최ㅗㅇ값을 구하는 함수

def rmse_val(predict_result,y) :
    return rmse(np.array(predict_result), np.array(y))

# 예측값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한번씩 대입

for i in range(len(x)) :
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제점수 = %.f, 예측점수=%.f" % (x[i], y[i],
    predict(x[i])))

