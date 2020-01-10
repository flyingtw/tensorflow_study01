import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

activation_func = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(activation_func, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print("%05d" % step, cost_val)
h, c, a = sess.run([hypothesis, activation_func, accuracy], feed_dict={X: x_data, Y: y_data})
print("\n[ 10,001번 학습 결과 ]")
print("1. 시그모이드 적용 : ", h.T)
print("2. + 활성함수 적용 :", [int(x) for x in c])
print("3. 기존 정답과 비교 : ", sum(y_data, []))
print("정확도 Accuracy :", a)
