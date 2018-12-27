# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#获取数据
mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

#构建模型，x接受输入，对连接权值w和偏置b初始值设为0
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#y表示输出（匹配概率），y是预测值，y_表示实际值，
#y = tf.nn.sigmoid(tf.matmul(x,w)+b)  #使用sigmoid作为整合函数
y=tf.nn.softmax(tf.matmul(x,w)+b)     #使用softmax函数整合函数
y_ = tf.placeholder("float", [None,10])

#定义损失函数loss，使用交叉熵作为损失函数，梯度下降法优化模型参数
#loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_cross_entropy = -tf.reduce_sum(y_*tf.log(y))
global_step = tf.Variable(0, trainable = False)

#定义衰减学习率
lrn_rate_base = 0.001
lrn_rate_decay = 0.9
lrn_rate_step = 600    #600轮更新一次学习率

learning_rate = tf.train.exponential_decay(lrn_rate_base,
                                           global_step,
                                           lrn_rate_step,
                                           lrn_rate_decay,
                                           staircase = True)

#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_cross_entropy,global_step = global_step)

#使用动量优化器
opt=tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)
train_step=opt.minimize(loss_cross_entropy,global_step=global_step)

#初始化我们创建的变量
init = tf.global_variables_initializer()

#执行运算初始化
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if(i%100 == 0):
        #模型评估
        correct_prediction = tf.math.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        rate = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        #计算所学习到的模型在测试数据集上面的正确率
        print("after {} steps, accuracy:{}".format(i+100,rate))