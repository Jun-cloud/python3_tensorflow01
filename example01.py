import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
# -100 ~ 100 사이의 랜덤 값
W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b
# cost : 비용
# reduce_mean : 평균 값; square : 제곱
cost = tf.reduce_mean(tf.square(H - Y))
# 경사하강 그래프에서 얼마만큼 이동(점프)할 지
a = tf.Variable(0.01)
# 경사 하강 라이브러리
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5001):
    sess.run(train, feed_dict={X: xData, Y: yData})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X: [8]}))