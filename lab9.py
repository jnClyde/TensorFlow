import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

L1 = 500
L2 = 250
x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.truncated_normal([784, L1], stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal([L1], stddev = 0.1))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev = 0.1))
b2 = tf.Variable(tf.truncated_normal([L2], stddev = 0.1))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W = tf.Variable(tf.truncated_normal([L2, 10], stddev = 0.1))
b = tf.Variable(tf.truncated_normal([10], stddev = 0.1))
y = tf.nn.softmax(tf.matmul(y2, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(4000):
    print(i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))    
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))    

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

saver = tf.train.Saver()
saver.save(sess, 'model.cpkt')
