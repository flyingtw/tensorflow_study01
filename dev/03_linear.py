import tensorflow as tf

X = [1, 2, 4, 5, 6, 7]
Y = [10, 20, 40, 50, 60, 70]

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = W * X + b
# cost(오차)  즉, 오차가 가장 적은값을 찾아야 한다.(제곱해서 양수화 함)
# reduce_mean (모든 오차의 제곱 값을 더하여 평균 내는 과정)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최저 Cost 를 찾는 방법(미분해서 기울기 구한후 최저값 찾는등의 복잡한 수식을 계산)
# learning_rate = Gradient Descent 알고리즘의 일정 간격으로 내려가는 간격
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Variable  변수에 값 넣는것
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 2001번 학습 진행
for step in range(2001):
    sess.run(train)
    if step % 200 == 0: # 2001번의 학습중 200단위로 끊어 출력
        print(step, sess.run(cost), sess.run(W), sess.run(b))
