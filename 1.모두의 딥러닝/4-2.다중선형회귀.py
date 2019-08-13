# 다중 선형 회귀

# 독립변수 x1, x2 다중 회귀를 경사하강법으로 실행

import tensorflow as tf

# x1, x2, y의 데이터값
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]     # 새로 추가되는 데이터
y_data = [y_row[2] for y_row in data]

learning_rate = 0.1

a1 = tf.Variable(tf.random_uniform([1],0,10, dtype = tf.float64, seed = 0))
a2 = tf.Variable(tf.random_uniform([1],0,10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0))

y = a1 * x1 +a2 * x2 + b

rmse = tf.sqrt(tf.reduce_mean(tf.square( y -y_data)))

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        sess.run(gradient_decent)
        if step % 100 == 0 :
            print("Epoch: %.f, Rmse = %.f, 기울기 a1 = %.f, 기울기 a2 = %.f, y절편 b = %.f" % (step, sess.run(rmse), sess.run(a1),
                                                                                                                     sess.run(a2),sess.run(b)))


for i in range(4) :
    y = 1.2301 * x1[i] + 2.1633 * x2[i] + 77.8117
