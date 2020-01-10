import tensorflow as tf

box = tf.constant("Hello world!")
print(box)
sess = tf.Session()
print(sess.run(box))
