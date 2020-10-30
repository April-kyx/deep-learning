import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 自己定义一个神经网络隐藏层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    WX_plus_biase = tf.matmul(inputs, Weight) + biase
    if activation_function is None:
        outputs = WX_plus_biase
    else:
        outputs = activation_function(WX_plus_biase)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784], name='x_inputs')
ys = tf.placeholder(tf.float32, [None, 10], name='y_inputs')

# l1 = add_layer(xs, 784, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
