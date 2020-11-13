import tensorflow as tf

with tf.device("/cpu: 0"):
    a = tf.Variable(3.0)
    b = tf.Variable(4.0)

c = a * b

config = tf.ConfigProto()
config.log_device_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
print(sess.run(c))