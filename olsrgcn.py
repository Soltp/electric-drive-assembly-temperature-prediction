import tensorflow as tf
#RNNCell=tf.compat.v1.nn.rnn_cell.BasicRNNCell   
from tensorflow.contrib.rnn import RNNCell  
from utils_RGCN_WP import calculate_laplacian

class tgcnCell(RNNCell):
    

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
        self._adj.append(calculate_laplacian(adj)) 



    @property
    def state_size(self):
        return self._nodes1 * self._units

    @property
    def output_size(self):
        return self._units
   
    def __call__(self, inputs, state, scope=None):  
        with tf.variable_scope(scope or "tgcn"):  
            with tf.variable_scope("gates"):  
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope)) 
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)  
                # [r, u] = sigmoid(W[x, h] + b)
            with tf.variable_scope("candidate"): 
                r_state = r * state  # r * h
                # c = tanh(W[x, (r * h)] + b)
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
            # h := u * h + (1 - u) * c
        return new_h, new_h

    
    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):

        inputs1 = tf.transpose(inputs, perm=[1, 0])
  
        state = tf.reshape(state, (-1, self._nodes1, self._units))


        scope = tf.get_variable_scope()  
        with tf.variable_scope(scope):
           
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
             weights1 = tf.get_variable('weights1', [input_size, output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
             x4 = tf.matmul(x2, weights1)  # w1*tanh(AXW)
             # x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)  
             biases = tf.get_variable(
                 "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
             # w1*tanh(AXW)+b
             x5 = tf.nn.bias_add(x4, biases)  
             # x = self._act(tf.nn.bias_add(x, biases))
             x5 = tf.reshape(x5, shape=[-1, self._nodes1, output_size])
             x5 = tf.reshape(x5, shape=[-1, self._nodes1 * output_size]) 
        return x5
