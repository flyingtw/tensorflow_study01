import tensorflow as tf

# 숫자 뒤에 . 을 붙임으로써 실수형임을 명시
# x_data : 1차원 벡터
x_data = [[10., 20.],
          [30., 12.],
          [25., 35.],
          [32., 40.]]
# y_data : 2차원 벡터
y_data = [[5.],
          [7.],
          [10.],
          [12.]]



X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(6001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 1500 == 0:
        print("오차 크기 :", cost_val, "\n기존 값에 대한 예측 :\n", hy_val, "\n")

print("중간고사 20점 / 기말고사 30점 일때, 장학금은? :", sess.run(hypothesis, feed_dict={X: [[20., 30.]]}))
print("중간고사 40점 / 기말고사 70점 일때, 장학금은? :", sess.run(hypothesis, feed_dict={X: [[40., 70.]]}))
