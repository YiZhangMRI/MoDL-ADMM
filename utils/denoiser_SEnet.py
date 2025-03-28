import tensorflow as tf
import scipy.io as scio
from utils.operator import pre_padding


class GlobalAvgPool_3D(tf.keras.layers.Layer):
    """
    3D global Avg pooling per frame on 3D images
    """

    def __init__(self):
        super(GlobalAvgPool_3D, self).__init__()
        self.AvgPool = tf.keras.layers.GlobalAveragePooling3D()
        # self.AvgPool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        nb, nf, nx, ny, nc = x.shape
        # x = tf.reshape(x, [nb * nf, nx, ny, nc])
        x = self.AvgPool(x)
        return tf.reshape(x, [nb, 1, 1, 1, nc])
        # return tf.reshape(x, [nb, nf, 1, 1, nc])


class SE_Layer_3(tf.keras.layers.Layer):
    """
    U_a: Conv-3D 3x3 stride 1
    """

    def __init__(self, n_f=32, kernel_initializer=tf.keras.initializers.GlorotNormal(), lastLayer=False, layer_no=1):
        super(SE_Layer_3, self).__init__()
        self.nf = n_f
        self.kernel_size = 3
        self.kernel_initializer = kernel_initializer
        self.lastLayer = lastLayer
        self.layer_no = layer_no
        self.pad_width = tf.math.floor(self.kernel_size / 2)
        self.padding_size_a = [self.pad_width] * 3
        self.Conv_a = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='valid', use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer='zeros')
        self.ReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.AvgPool = GlobalAvgPool_3D()
        self.W_a = tf.keras.layers.Dense(self.nf, activation=None, use_bias=False,
                                         kernel_initializer=self.kernel_initializer)
        # self.W_r = tf.keras.layers.Dense(self.nf/2, activation=self.ReLU, use_bias=False,
        #                                  kernel_initializer=self.kernel_initializer)
        # when using 2 selective kernels, sigmoid is equivalent to softmax
        self.Softmax = tf.keras.layers.Activation('sigmoid')

    def call(self, input, training=True):
        input_a = pre_padding(input, pad_size=self.padding_size_a, axis=[1, 2, 3], mode="REFLECT")
        # tf.print(input_a.shape)
        if self.lastLayer:
            u_a = self.Conv_a(input_a)
        else:
            u_a = self.ReLU(self.Conv_a(input_a))
            # tf.print(u_a.shape)
            # quit()
        a = self.Softmax(self.W_a(self.AvgPool(u_a)))  # no hidden layer
        # a = self.Softmax(self.W_a(self.W_r(self.AvgPool(u_a))))  # hidden layer with r=2

        if self.lastLayer:
            return tf.cast(u_a * a, dtype='float32')
        return u_a * a


class SE_Layer_5(tf.keras.layers.Layer):
    """
    U_b: Conv-3D 3x3 stride 1 dilation_rate 2,2,1 (5x5)
    """

    def __init__(self, n_f=32, kernel_initializer=tf.keras.initializers.GlorotNormal(), lastLayer=False, layer_no=1):
        super(SE_Layer_5, self).__init__()
        self.nf = n_f
        self.kernel_size = 3
        self.dilation_rate = [1, 2, 2]
        self.kernel_initializer = kernel_initializer
        self.lastLayer = lastLayer
        self.layer_no = layer_no
        self.pad_width = tf.math.floor(self.kernel_size / 2)
        self.padding_size_b = [self.pad_width * i for i in self.dilation_rate]
        self.Conv_b = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, dilation_rate=self.dilation_rate,
                                             padding='valid', use_bias=True, kernel_initializer=self.kernel_initializer,
                                             bias_initializer='zeros')
        self.ReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.AvgPool = GlobalAvgPool_3D()
        self.W_a = tf.keras.layers.Dense(self.nf, activation=None, use_bias=False,
                                         kernel_initializer=self.kernel_initializer)
        # self.W_r = tf.keras.layers.Dense(self.nf/2, activation=self.ReLU, use_bias=False,
        #                                  kernel_initializer=self.kernel_initializer)
        # when using 2 selective kernels, sigmoid is equivalent to softmax
        self.Softmax = tf.keras.layers.Activation('sigmoid')

    def call(self, input, training=True):
        input_b = pre_padding(input, pad_size=self.padding_size_b, axis=[1, 2, 3], mode="REFLECT")
        # tf.print(input_b.shape)
        if self.lastLayer:
            u_b = self.Conv_b(input_b)
        else:
            u_b = self.ReLU(self.Conv_b(input_b))
            # tf.print(u_b.shape)
            # quit()
        a = self.Softmax(self.W_a(self.AvgPool(u_b)))  # no hidden layer
        # a = self.Softmax(self.W_a(self.W_r(self.AvgPool(u_b))))  # hidden layer with r=2

        if self.lastLayer:
            return tf.cast(u_b * a, dtype='float32')
        return u_b * a


class Nw_SE(tf.keras.layers.Layer):
    """
    Nw block using Squeeze Excitation Networks i.e. SENet
    """

    def __init__(self, n_l=5, n_f=32, n_k=3, kernel_initializer=tf.keras.initializers.GlorotNormal()):
        super(Nw_SE, self).__init__()
        self.n_l = n_l
        self.n_f_list = [n_f] * (n_l - 1) + [2]
        self.kernel_initializer = kernel_initializer
        self.cell_list = []
        if n_k == 3:
            tf.print('SE3')
            for l in range(self.n_l):
                if l == self.n_l - 1:
                    self.cell_list.append(SE_Layer_3(2, self.kernel_initializer, lastLayer=True, layer_no=self.n_l))
                else:
                    self.cell_list.append(SE_Layer_3(self.n_f_list[l], self.kernel_initializer, lastLayer=False,
                                                     layer_no=l + 1))
        else:
            tf.print('SE5')
            for l in range(self.n_l):
                if l == self.n_l - 1:
                    self.cell_list.append(SE_Layer_5(2, self.kernel_initializer, lastLayer=True, layer_no=self.n_l))
                else:
                    self.cell_list.append(SE_Layer_5(self.n_f_list[l], self.kernel_initializer, lastLayer=False,
                                                     layer_no=l + 1))

    def call(self, input, training=True):
        nw = {'c_' + str(0): input}
        for l in range(0, self.n_l):
            # tf.print('c_' + str(l), nw['c_' + str(l)].shape)
            # tf.print('c_' + str(l), nw['c_' + str(l)].dtype)
            nw['c_' + str(l + 1)] = self.cell_list[l](nw['c_' + str(l)], training=training)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].shape)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].dtype)
        return nw['c_' + str(l + 1)]
