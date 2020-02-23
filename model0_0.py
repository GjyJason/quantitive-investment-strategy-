import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from data0_0 import data
import gc
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

class ml_model(data):
    def __init__(self, index=['000001.XSHG', '000002.XSHG', '000016.XSHG',
                              '000903.XSHG', '000904.XSHG', '000905.XSHG',
                              '000906.XSHG', '000907.XSHG', '399001.XSHE', '399004.XSHE',
                              '399005.XSHE', '399006.XSHE', '399009.XSHE',
                              '399010.XSHE', '399011.XSHE', '399310.XSHE', '399310.XSHE',
                              '399310.XSHE', '000852.XSHG', '000300.XSHG'], start_date='2012-05-31', end_date='2019-05-31', frequency='daily'):
        data.__init__(self, index, start_date, end_date, frequency)
        self.batch_size = 64
        self.saver = None

        # 模型超参数
        self.num_units = 96   # lstm单元隐藏层细胞数量，即Wx和Wh的行数
        self.learning_rate = 0.005  # 学习率
        self.epoch = 50   # 训练轮数
        self.numda = 0.0001        # L2正则化系数





    # lstm网路单元
    def lstm_model(self):
        with tf.variable_scope(name_or_scope='lstm_model', reuse=tf.AUTO_REUSE):
            # 输入
            # 特征数量为25，batch_size，seq_length并不固定
            x = tf.placeholder(shape=(None, None, 25), name='input', dtype=tf.float64)
            print(x.name)
            # 占位符，说明是否正在训练（影响batch_normalization层）
            is_training = tf.placeholder(name='is_training', dtype=tf.bool)

            # lstm构造
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, initializer=tf.orthogonal_initializer(),
                                                    num_proj=48, reuse=tf.AUTO_REUSE,
                                                name='lstm_cell', activation='sigmoid')
            #lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, initializer=tf.random_normal_initializer(),
            #num_proj = 48, reuse = tf.AUTO_REUSE,
            #name = 'lstm_cell', activation = 'sigmoid')
            c = (tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=tf.float64))[0][:, -1, :]

        with tf.variable_scope(name_or_scope='fully_connection', reuse=tf.AUTO_REUSE):
            # 全连接层
            b_1 = tf.get_variable(name='bias_1', dtype=tf.float64, initializer=tf.constant(value=np.zeros((16,)), dtype=tf.float64))
            w_1 = tf.get_variable(name='weight_1', shape=(48, 16), dtype=tf.float64,
                                initializer=initializers.xavier_initializer())

            b_2 = tf.get_variable(name='bias_2', dtype=tf.float64, initializer=tf.constant(value=np.zeros((2,)), dtype=tf.float64))
            w_2 = tf.get_variable(name='weight_2', shape=(16, 2), dtype=tf.float64,
                                  initializer=initializers.xavier_initializer())
            #w = tf.get_variable(name='weight', shape=(32, 2), dtype=tf.float64,
                                 #initializer=tf.random_normal_initializer())

            c2 = tf.matmul(c, w_1) + b_1
            # batch normalization层
            batch_norm = tf.layers.batch_normalization(inputs=tf.matmul(c2, w_2) + b_2, training=is_training, name='batch_norm',
                                                       epsilon=0)
            # 激活函数加softmax输出概率 [P涨，P跌]
            # tanh_output = tf.tanh(batch_norm, 'tanh')
            # output = tf.nn.softmax(tanh_output)
            leaky_relu_output = tf.nn.leaky_relu(features=batch_norm, alpha=0.2, name='relu')
            output = tf.nn.softmax(logits=leaky_relu_output, name='output')


            # 返回模型输出值,和占位符
            return output, x, is_training








    # 训练
    def train(self):
        train_graph = tf.Graph()
        with train_graph.as_default():
            output, x, is_training = self.lstm_model()
            # 训练集的标签
            y = tf.placeholder(shape=(None, ), name='label', dtype=tf.float64)
            print(y.name)
            # 参数L2正则化减少过拟合
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.numda), tf.trainable_variables())
            # 损失函数计算（交叉熵加L2正则化）
            cse = tf.convert_to_tensor(value=0, dtype=tf.float64)
            for i in range(self.batch_size):
                cse = cse - 0.61022*tf.multiply(y[i], tf.log(output[i][0])) - 2.76825*tf.multiply(1 - y[i], tf.log(output[i][1]))
            cse = cse/self.batch_size
            cse = cse + reg

            # 准确率计算
            acc_num = tf.convert_to_tensor(0)
            n = tf.convert_to_tensor(0)
            for i in range(self.batch_size):
                acc1 = tf.convert_to_tensor(0)
                acc2 = tf.convert_to_tensor(0)
                #acc3 = tf.convert_to_tensor(0)
                n = tf.cond(pred=tf.equal(y[i], 0),
                               true_fn=lambda: tf.add(n, 1), false_fn=lambda: tf.add(n, 0))
                acc1 = tf.cond(pred=tf.logical_and(tf.equal(y[i], 1),
                                                   tf.greater(output[i][0], output[i][1])
                                                                ),
                               true_fn=lambda: tf.add(acc1, 1), false_fn=lambda: tf.add(acc1, 0))
                acc2 = tf.cond(pred=tf.logical_and(tf.equal(y[i], 0),
                                                   tf.greater(output[i][1], output[i][0])
                                                                  ),
                               true_fn=lambda: tf.add(acc2, 1), false_fn=lambda: tf.add(acc2, 0))
                acc_num = acc_num + acc1 + acc2
                #acc_num = acc_num + acc2
            acc = acc_num / self.batch_size
            #acc = acc_num / n
                #acc3 = tf.cond(pred=tf.logical_and(tf.equal(y[i][2], 1),
                  #                                 tf.logical_and(tf.greater(output[i][2], output[i][0]),
                 #                                                 tf.greater(output[i][2], output[i][1]))),
                 #              true_fn=lambda: tf.add(acc3, 1), false_fn=lambda: tf.add(acc3, 0))
                #acc_num = acc_num + acc1 + acc2 + acc3


            # 批量梯度下降(Adam?)优化损失函数，好像Adam更好一些
            optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_1 = optimizer_1.minimize(loss=cse)
                train_op_2 = optimizer_2.minimize(loss=cse)

            '''
            var_list = []
            list_2 = []
            for v in tf.global_variables():
                if v in tf.trainable_variables():
                    var_list.append(v)
                else:
                    list_2.append(v)
            saver = tf.train.Saver(var_list=var_list)
            '''

            saver = tf.train.Saver()
            # 开启session，载入最后报存的模型\初始化模型
            sess = tf.Session()

            # 以下两行可用作开启第一次训练与延续上一次训练之间切换
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path="C:\\Users\\Jason\\Desktop\\hengqin quant finance\\saved_model1_0\\lstm_model-00.3")
            #sess.run(tf.variables_initializer(list_2))


            for i in range(self.epoch):
                # 获取minibatch
                batch_x, batch_y = self.get_mini_batch(self.batch_size)

                # 训练，返回损失函数值
                if i == 0:
                    print(sess.run((cse, sess.graph.get_tensor_by_name('fully_connection/relu:0'), output), {x: batch_x, y: batch_y, is_training: True}))
                    sess.run(fetches=train_op_2, feed_dict={x: batch_x, y: batch_y, is_training: True})
                else:
                    print(sess.run((cse, acc), {x: batch_x, y: batch_y, is_training: True}))
                    sess.run(fetches=train_op_2, feed_dict={x: batch_x, y: batch_y, is_training: True})
            saver.save(sess=sess,
                         save_path='C:\\Users\\Jason\\Desktop\\hengqin quant finance\\saved_model1_0\\lstm_model-00.3')
            #saver.save(sess=sess,
             #          save_path='./lstm_model-00.1')



            '''
            builder = tf.saved_model.builder.SavedModelBuilder('./lstm_saved_model')
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
            builder.save()
            '''
            varlist = [v.name[:-2] for v in tf.global_variables()]
            varlist.append('fully_connection/output')
            varlist.append('lstm_model/input')
            varlist.append('label')
            varlist.append('lstm_model/is_training')

            print(varlist)

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, varlist)
            with tf.gfile.FastGFile('./lstm_model_5.pb', mode='wb') as f:

                f.write(constant_graph.SerializeToString())

            sess.close()

    # 测试
    def test(self, x_test, y_test):

        test_graph = tf.Graph()
        with test_graph.as_default():
            output, x, is_training = self.lstm_model()
            # 测试集的标签
            y = tf.placeholder(shape=(None,), name='label', dtype=tf.float64)
            # 准确率计算
            acc_num = tf.convert_to_tensor(0)
            n = tf.convert_to_tensor(0)
            for i in range(len(x_test)):
                acc1 = tf.convert_to_tensor(0)
                acc2 = tf.convert_to_tensor(0)
                # acc3 = tf.convert_to_tensor(0)
                n = tf.cond(pred=tf.equal(y[i], 0),
                            true_fn=lambda: tf.add(n, 1), false_fn=lambda: tf.add(n, 0))
                acc1 = tf.cond(pred=tf.logical_and(tf.equal(y[i], 1),
                                                   tf.greater(output[i][0], output[i][1]),
                                                   ),
                               true_fn=lambda: tf.add(acc1, 1), false_fn=lambda: tf.add(acc1, 0))
                acc2 = tf.cond(pred=tf.logical_and(tf.equal(y[i], 0),
                                                   tf.greater(output[i][1], output[i][0])
                                                                  ),
                               true_fn=lambda: tf.add(acc2, 1), false_fn=lambda: tf.add(acc2, 0))
                #acc_num = acc_num + acc1 + acc2
                acc_num = acc_num + acc2
            #acc = acc_num/len(x_test)
            acc = acc_num/n

            saver = tf.train.Saver()
            # 开启session,载入最新模型
            sess = tf.Session()
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path="C:\\Users\\Jason\\Desktop\\hengqin quant finance\\saved_model1_0\\lstm_model-00.3")

            print(sess.run(fetches=(acc, output, sess.graph.get_tensor_by_name('fully_connection/relu:0')),
                           feed_dict={x: x_test, y: y_test, is_training: False}))


        '''
        sess = tf.Session()
        with gfile.FastGFile('./lstm_model_1.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            print(sess.run(fetches=(sess.graph.get_tensor_by_name('fully_connection/output:0')),
                           feed_dict={sess.graph.get_tensor_by_name('lstm_model/input:0'): x_test,
                                      sess.graph.get_tensor_by_name('label:0'): y_test,
                                      sess.graph.get_tensor_by_name('lstm_model/is_training:0'): False}))
        '''
    # 预测
    def predict(self, x_predict):
        predict_graph = tf.Graph()
        with predict_graph.as_default():
            output, x, is_training = self.lstm_model()

            saver = tf.train.Saver()
            # 开启session
            sess = tf.Session()
            saver.restore(sess=sess, save_path="C:\\Users\\Jason\\Desktop\\hengqin quant finance\\saved_model1_0\\lstm_model-00.1")
            o = sess.run(fetches=output, feed_dict={x: x_predict, is_training: False})
            sess.close()
            print(o)
            return o

model = ml_model(index=['000001.XSHG', '000002.XSHG', '000016.XSHG',
                              '000903.XSHG', '000904.XSHG', '000905.XSHG',
                              '000906.XSHG', '000907.XSHG', '399001.XSHE', '399004.XSHE',
                              '399005.XSHE', '399006.XSHE', '399009.XSHE',
                              '399010.XSHE', '399011.XSHE', '399310.XSHE', '399310.XSHE',
                              '399310.XSHE', '000852.XSHG', '000300.XSHG'], start_date='2012-01-01', end_date='2018-01-01', frequency='daily')

model.get_data()
model.divide_data()

model.train()

#model.test(x_test=model.x_test, y_test=model.y_test)
