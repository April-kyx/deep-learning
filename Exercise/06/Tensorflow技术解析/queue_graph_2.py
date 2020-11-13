import tensorflow as tf

q = tf.FIFOQueue(capacity=1000, dtypes=tf.float32)
counter = tf.Variable(0.0)
increment_op = tf.assign_add(counter, tf.constant(1.0))  # 操作：给计数器加1
enqueue_op = q.enqueue([counter])

# 创建队列管理器
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

# 使用协调器(coordinator)来管理线程
sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()

# 启动入队线程，协调器是线程的参数
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

coord.request_stop()  # 通知其他线程关闭

# 主线程
for i in range(0, 10):
    try:
        print(sess.run(q.dequeue()))
    except tf.errors.OutOfRangeError:
        break

coord.join(enqueue_threads)
