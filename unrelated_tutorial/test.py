# Just disables the warning, doesn't enable AVX/FMA
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b

y = tf.placeholder(tf.float32)

# loss
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

# optimize model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x:[0,1,2,3], y:[0,-1,-2,-3]})

# print(sess.run(loss, {x:[0,1,2,3], y:[0,-1,-2,-3]}))

print(sess.run([W, b]))

