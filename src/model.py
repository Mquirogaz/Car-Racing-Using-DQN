import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class QNetwork(object):
    def __init__(self, input_size, output_size, name):
        self.name = name

    def weight_variable(self, shape, fanin=0):
        if fanin == 0:
            initial = tf.random.truncated_normal(shape, stddev=0.01)
        else:
            mod_init = 1.0 / np.sqrt(fanin)
            initial = tf.random.uniform(shape, minval=-mod_init, maxval=mod_init)

        return tf.Variable(initial)

    def bias_variable(self, shape, fanin=0):
        if fanin == 0:
            initial = tf.constant(0.01, shape=shape)
        else:
            mod_init = 1.0 / np.sqrt(fanin)
            initial = tf.random.uniform(shape, minval=-mod_init, maxval=mod_init)

        return tf.Variable(initial)

    def variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def copy_to(self, dst_net):
        v1 = self.variables()
        v2 = dst_net.variables()

        for i in range(len(v1)):
            v2[i].assign(v1[i]).eval()

    def print_num_of_parameters(self):
        list_vars = self.variables()
        total_parameters = 0
        for variable in list_vars:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('# of parameters in network', self.name, ':', total_parameters, ' -> ', np.round(float(total_parameters) / 1000000.0, 2), 'M')

class QNetworkDueling(QNetwork):
    def __init__(self, input_size, output_size, name):
        super().__init__(input_size, output_size, name)

        self.W_conv1 = self.weight_variable(shape=[8, 8, 4, 32])
        self.B_conv1 = self.bias_variable(shape=[32])
        self.stride1 = 4

        self.W_conv2 = self.weight_variable(shape=[4, 4, 32, 64])
        self.B_conv2 = self.bias_variable(shape=[64])
        self.stride2 = 2

        self.W_conv3 = self.weight_variable(shape=[3, 3, 64, 64])
        self.B_conv3 = self.bias_variable(shape=[64])
        self.stride3 = 1

        self.W_fc4a = self.weight_variable(shape=[7 * 7 * 64, 512])
        self.B_fc4a = self.bias_variable(shape=[512])

        self.W_fc4b = self.weight_variable(shape=[7 * 7 * 64, 512])
        self.B_fc4b = self.bias_variable(shape=[512])

        self.W_fc5a = self.weight_variable(shape=[512, output_size])
        self.B_fc5a = self.bias_variable(shape=[output_size])

        self.W_fc5b = self.weight_variable(shape=[512, output_size])
        self.B_fc5b = self.bias_variable(shape=[output_size])

    def __call__(self, input_tensor):
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        self.h_conv1 = tf.nn.relu(tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, self.stride1, self.stride1, 1],
                                                padding='VALID') + self.B_conv1)
        self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, self.stride2, self.stride2, 1],
                                                padding='VALID') + self.B_conv2)
        self.h_conv3 = tf.nn.relu(tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, self.stride3, self.stride3, 1],
                                                padding='VALID') + self.B_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7 * 7 * 64])

        self.h_fc4a = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4a) + self.B_fc4a)
        self.h_fc4b = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4b) + self.B_fc4b)

        self.h_fc5a_value = tf.matmul(self.h_fc4a, self.W_fc5a) + self.B_fc5a
        self.h_fc5b_advantage = tf.matmul(self.h_fc4b, self.W_fc5b) + self.B_fc5b

        self.h_fc6 = self.h_fc5a_value + (self.h_fc5b_advantage - tf.reduce_mean(self.h_fc5b_advantage, axis=[1,], keepdims=True))

        return tf.squeeze(self.h_fc6, axis=0)
