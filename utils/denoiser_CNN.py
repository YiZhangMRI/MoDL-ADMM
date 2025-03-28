import tensorflow as tf
import numpy as np
from utils.operator import complex2c, pre_padding


class Conv_Layer(tf.keras.layers.Layer):
    """
    Conv-3D + BN + ReLU
    last: 2ch-Conv-3D + BN
    """

    def __init__(self, n_f=64, kernel_size=3, kernel_initializer=tf.keras.initializers.GlorotNormal(), lastLayer=False):
        super(Conv_Layer, self).__init__()
        self.nf = n_f
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.lastLayer = lastLayer
        if self.lastLayer:
            # self.Conv = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='same', use_bias=True,
            self.Conv = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='valid', use_bias=True,
                                               kernel_initializer=self.kernel_initializer,
                                               bias_initializer='zeros')
        else:
            # self.Conv = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='same', use_bias=True,
            self.Conv = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='valid', use_bias=True,
                                               kernel_initializer=self.kernel_initializer,
                                               bias_initializer='zeros')
            self.ReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
        # self.BN = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
        #                                              beta_initializer='zeros', gamma_initializer='ones')

    def call(self, input, training=True):
        input = pre_padding(input, pad_size=[tf.math.floor(self.kernel_size/2)]*3, axis=[1, 2, 3], mode="REFLECT")
        if self.lastLayer:
            # return tf.cast(self.BN(self.Conv(input), training=training), dtype='float32')
            return tf.cast(self.Conv(input), dtype='float32')
        else:
            # return self.ReLU(self.BN(self.Conv(input), training=training))
            return self.ReLU(self.Conv(input))


class Nw_CNN(tf.keras.layers.Layer):
    """
    Nw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: It is the output of residual learning after adding the input back.
    """

    def __init__(self, n_l=5, n_f=64, kernel_size=3, kernel_initializer=tf.keras.initializers.GlorotNormal()):
        super(Nw_CNN, self).__init__()
        self.n_l = n_l
        self.n_f = n_f
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.cell_list = []

    def build(self, input_shape):
        # tf.print("layer built with input_shape", input_shape)
        for l in range(self.n_l):
            if l == self.n_l - 1:
                self.cell_list.append(Conv_Layer(2, self.kernel_size, self.kernel_initializer, lastLayer=True))
            else:
                self.cell_list.append(Conv_Layer(self.n_f, self.kernel_size, self.kernel_initializer, lastLayer=False))

    def call(self, input, training=True):
        nw = {'c_' + str(0): input}
        for l in range(0, self.n_l):
            # tf.print('c_' + str(l), nw['c_' + str(l)].shape)
            # tf.print('c_' + str(l), nw['c_' + str(l)].dtype)
            nw['c_' + str(l + 1)] = self.cell_list[l](nw['c_' + str(l)], training=training)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].shape)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].dtype)
        return nw['c_' + str(l + 1)]
