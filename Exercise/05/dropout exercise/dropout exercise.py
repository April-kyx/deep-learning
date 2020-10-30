import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)

# 自己定义一个神经网络隐藏层函数
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    WX_plus_biase = tf.matmul(inputs, Weight) + biase
    WX_plus_biase = tf.nn.dropout(WX_plus_biase, keep_prob)
    if activation_function is None:
        outputs = WX_plus_biase
    else:
        outputs = activation_function(WX_plus_biase)
    tf.summary.histogram(layer_name + './outputs', outputs)
    return outputs

keep_prob = tf.placeholder(tf.float32) #我们需要设置要有多少概率的神经网络不被dropout
xs = tf.placeholder(tf.float32, [None, 64]) #8*8
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh) #隐藏层
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax) #输出层

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    # 文件记录
    train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./logs/test", sess.graph)
    sess.run(init)

    for i in range(500):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})

        if i % 50 == 0:
            # 记录损失
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
            test_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i )