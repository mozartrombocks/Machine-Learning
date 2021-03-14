import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.disable_eager_execution()
x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)

# sess = tf.compat.v1.Session()
# print(sess.run(result))
# sess.close()

with tf.compat.v1.Session() as sess:
    output = sess.run(result)
    print(sess.run(result))

print(output)

