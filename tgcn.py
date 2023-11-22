import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from utils_TGCN import calculate_laplacian

class tgcnCell(RNNCell):

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes,num_nodes1, act=tf.nn.tanh, reuse=None):

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
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        inputs1 = tf.transpose(inputs, perm=[1, 0])
        state = tf.reshape(state, (-1, self._nodes1, self._units))
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
             weights11 = tf.get_variable('weights1', [self._nodes, self._nodes],
                                         initializer=tf.contrib.layers.xavier_initializer())
             for m in self._adj:
                 x1 = tf.matmul(m, weights11)
             x2 = tf.matmul(x1, inputs1)
             x3 = tf.transpose(x2, perm=[1, 0])
             x4 = tf.expand_dims(x3, 2)
             x_s = tf.concat([x4, state], axis=2)
             input_size = x_s.get_shape()[2].value
             x5 = tf.reshape(x_s, shape=[-1, input_size])
             weights1 = tf.get_variable('weights2', [input_size, output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
             x6 = tf.matmul(x5, weights1)
             biases = tf.get_variable(
                 "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
             x7 = tf.nn.bias_add(x6, biases)
             x8 = tf.reshape(x7, shape=[-1, self._nodes1, output_size])
             x9 = tf.reshape(x8, shape=[-1, self._nodes1 * output_size])
        return x9
