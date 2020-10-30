import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 自己定义一个神经网络隐藏层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([out_size, in_size]))
    biase = tf.Variable(tf.zeros([out_size, 1]) + 0.1)
    WX_plus_biase = tf.matmul(Weight, inputs) + biase
    if activation_function is None:
        outputs = WX_plus_biase
    else:
        outputs = activation_function(WX_plus_biase)
    return outputs


# 定义好所以的数据形式(定好维度)
x_data = np.linspace(-1, 1, 300)[np.newaxis, :]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 构建神经网络
x_hot = tf.placeholder(tf.float32, shape=[1, None], name="x")
y_hot = tf.placeholder(tf.float32, shape=[1, None], name="y")

layer1 = add_layer(x_hot, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - y_hot), reduction_indices=[0]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #数据可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()


    for i in range(1000):
        sess.run(train_step, feed_dict={x_hot: x_data, y_hot: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={x_hot:x_data, y_hot:y_data}))
            # 数据可视化
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={x_hot: x_data})
            lines = ax.plot(np.squeeze(x_data), np.squeeze(prediction_value), 'r-', lw=5)
            plt.pause(0.1)
    plt.pause(0)