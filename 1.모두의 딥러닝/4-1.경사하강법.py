# 4. 코딩으로 확인하는 최소제곱




########## 경사하강법 ##########

import tensorflow as tf

# x,y의 데이터 값
data = [[2,81], [4,93], [6, 91], [8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]



a = tf.Variable(tf.random_uniform([1],0,10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0))

y = a * x_data + b

rmse = tf.sqrt(tf.reduce_mean(tf.square( y -y_data)))

# 학습율 값

learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(5001) :
        sess.run(gradient_decent)
        if step % 100 == 0 :
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % (step,sess.run(rmse),sess.run(a),sess.run(b)))