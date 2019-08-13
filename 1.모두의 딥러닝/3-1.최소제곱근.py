############ 최소 제곱근 오차 ############

import numpy as np

x = [2,4,6,8]
y = [81, 93, 91, 97]
mx = np.mean(x)
my = np.mean(y)

print("x의 평균값:", mx)
print("y의 평균값:", my)

# 기울기 공식의 분모
divisor = sum([(mx - i )**2 for i in x])
divisor

# 기울기 공식의 분자
def top(x, mx, y, my) :
    d = 0
    for i in range(len(x)) :
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)

print("기울기의 분모 :", divisor)
print("기얼기의 분자 :", dividend)

# 기울기(a)와 y절편(b) 구하기
a = dividend  / divisor
b = my - (mx*a)

print("기울기 a = ", a)
print("y 절편 b = ", b)
