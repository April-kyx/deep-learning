import tensorflow as tf

q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes=tf.float32)

# 开启一个会话，执行10次入队操作，8次出队操作
sess = tf.Session()
for i in range(0, 10):
    sess.run(q.enqueue(i))

for i in range(0, 8):
   print(sess.run(q.dequeue()))