from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])


# 定义权重函数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 初始化权重参数
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])


# 定义模型
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # 第一个全连接层
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    # 第二个全连接层
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x))
training_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# 定义一个存储路径
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 定义一个计数器， 为训练轮数计数，
global_step = tf.Variable(0, name="global_step", trainable=False)

# 声明完所有变量后，调用tf.train.Saver()，后面定义的变量不会被训练模型并存储
saver = tf.train.Saver()
non_storable_variable = tf.Variable(777)  # 这个变量是不会被存储的

init = tf.global_variables_initializer()

# 训练模型并存储
with tf.Session() as sess:
    sess.run(init)

    start = global_step.eval()  # 得到global_step的初始值
    print("Start from:", start)

    for i in range(start, 100):
        # 以128作为batch_size
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
            sess.run(training_op, feed_dict={X: trX[start: end], Y: trY[start: end],
                                             p_keep_input: 0.8, p_keep_hidden: 0.5})

        global_step.assign(i).eval()  # 更新计数器
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)  # 存储模型

ckpt = tf.train.get_checkpoint_state(ckpt_dir)
