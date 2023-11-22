# -*- coding: utf-8 -*-
# 此文件只定义了一个TGCN计算单元的类
#20220330 修改 验证有效
#import numpy as np
# GCN 利用一般GCN 单层Y=tanh（AX1W1）；TGCN时序预测[r, u] = sigmoid(WY+b) ；c = tanh(WY+b)；h=u*state+（1-u）*c；
#权重不同相乘 Fin_OLS_GCNV11+Fin_olsgcnV22+utils2
import tensorflow as tf
#RNNCell=tf.compat.v1.nn.rnn_cell.BasicRNNCell   # tensorflow2.0以上写法
from tensorflow.contrib.rnn import RNNCell  ##tensorflow2.0以下写法
from utils_RGCN_WP import calculate_laplacian

class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes,num_nodes1, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._nodes1 = num_nodes1
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))  # 多了一个求解邻接矩阵



    @property
    def state_size(self):
        return self._nodes1 * self._units

    @property
    def output_size(self):
        return self._units
    # 重点部分  将图卷积结果放入GRU模块中进行输出
    def __call__(self, inputs, state, scope=None):  # 参数中state对应论文中上一时刻的状态，即ht-1

        with tf.variable_scope(scope or "tgcn"):  # variable_scope使得多个变量得以有相同的命名
            with tf.variable_scope("gates"):  
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope)) # tf.nn.sigmoid语句为激活函数，用于进行图卷积GC
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)  # tf.split语句用于分割卷积后的张量，重置门r用于控制先前时刻状态信息的度量，上传门u用于控制上传到下一状态的信息度量
                # [r, u] = sigmoid(W[x, h] + b)
            with tf.variable_scope("candidate"):  #  candidate 候选集部分的c对应公式‘ct=tanh（wc[]+bc）’中函数最后返回最新状态ht
                r_state = r * state  # r * h
                # c = tanh(W[x, (r * h)] + b)
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
            # h := u * h + (1 - u) * c
        return new_h, new_h

    # 图卷积过程 AXW+b
    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
        inputs1 = tf.transpose(inputs, perm=[1, 0])
        # inputs = tf.expand_dims(inputs, 2)  # 函数开头对特征矩阵进行构建，使用expand_dims增加输入维度
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes1, self._units))


        scope = tf.get_variable_scope()  # get_variable_scope获取变量后，将得到的特征矩阵与邻接矩阵相乘。
        with tf.variable_scope(scope):
             ###开始AX
             # self._nodes1=self._nodes - 2
             # weights11 = tf.get_variable('weights11', [self._nodes, self._nodes], initializer=tf.contrib.layers.xavier_initializer())
             weights11 = tf.get_variable('weights11', [self._nodes1, self._nodes],
                                         initializer=tf.contrib.layers.xavier_initializer())
            # biases11 = tf.get_variable(
             #    "biases0", [self._nodes], initializer=tf.constant_initializer(bias, dtype=tf.float32))
             for m in self._adj:
                 # x10 = tf.matmul(m, x0)  #AWX
                 x01 = tf.multiply(m, weights11)
                # x01 = tf.nn.bias_add(x10, biases11)
             x1 = tf.matmul(x01, inputs1)
             #x1 = self._act(tf.matmul(x01, inputs1))  # tanh(AWX)
             x11 = tf.transpose(x1, perm=[1, 0])
             x111 = tf.expand_dims(x11, 2)
             x_s = tf.concat([x111, state], axis=2)
             input_size = x_s.get_shape()[2].value  # get_shape()
             x2 = tf.reshape(x_s, shape=[-1, input_size])

             # x0 = tf.transpose(x_s, perm=[1, 2, 0])  # 按照[1, 2, 0]顺序将x_s进行转置
             # x0 = tf.reshape(x0, shape=[self._nodes, -1])
             # x2 = tf.reshape(x1, shape=[self._nodes1, input_size,-1])
             # x2 = tf.transpose(x2,perm=[2,0,1])
             # x2 = tf.reshape(x2, shape=[-1, input_size])
             ####结束AX
             weights1 = tf.get_variable('weights1', [input_size, output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
             x4 = tf.matmul(x2, weights1)  # w1*tanh(AXW)
             # x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)  #x=AX0W,这里乘以W值
             biases = tf.get_variable(
                 "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
             # w1*tanh(AXW)+b
             x5 = tf.nn.bias_add(x4, biases)  # 在tf.nn.bias_add处激活得到两层GCN，对应公式f（x，A）=deta（ARelu（AXW0）W1）
             # x = self._act(tf.nn.bias_add(x, biases))
             x5 = tf.reshape(x5, shape=[-1, self._nodes1, output_size])
             x5 = tf.reshape(x5, shape=[-1, self._nodes1 * output_size])  # 最终返回输出值x。此函数经历了很多张量的形式转换，对应论文空间关系建模过程
        return x5
