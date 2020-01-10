import tensorflow as tf

"""
# 1)
# a와 b 의 값이 딱 정해져 있을때
a = tf.constant(3)
b = tf.constant(2)


sum = a + b
min = a - b
mul = a * b
dvi = a / b


print("1. 덧셈 ) A + B = ", sess.run(sum))
print("2. 뺄셈 ) A - B = ", sess.run(min))
print("3. 곱하기 ) A * B = ", sess.run(mul))
print("4. 나누기 ) A / B = ", sess.run(dvi))


"""
# 2)
# tensorflow 의 placeholder 함수를 통하여 int형 변수임을 선언
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

# add 도 a와 b랑 마찬가지로 텐서이다.
add = a + b

sess = tf.Session()

print(sess.run(a, feed_dict={a: 3}))
print(sess.run(add, feed_dict={a: 3, b: 4}))
print(sess.run(add, feed_dict={a: [1, 2], b: [3, 4]}))
print(sess.run(add, feed_dict={a: [[1, 2], [3, 4]], b: [[5, 6], [7, 8]]}))
# 텐서플로우 자체가 다차원 배열이다! 즉, 1차원 또는 2차원 배열 형태로 값을 입력 할 수 있다.
