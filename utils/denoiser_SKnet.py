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


class SK_Layer(tf.keras.layers.Layer):
    """
    U_a: Conv-3D 3x3 stride 1
    U_b: Conv-3D 3x3 stride 1 dilation_rate 2,2,1 (5x5)
    """

    def __init__(self, n_f=45, kernel_initializer=tf.keras.initializers.GlorotNormal(), lastLayer=False, layer_no=1):
        super(SK_Layer, self).__init__()
        self.nf = n_f
        self.kernel_size = 3
        self.dilation_rate = [1, 2, 2]
        self.kernel_initializer = kernel_initializer
        self.lastLayer = lastLayer
        self.layer_no = layer_no
        self.pad_width = tf.math.floor(self.kernel_size / 2)
        self.padding_size_a = [self.pad_width] * 3
        self.padding_size_b = [self.pad_width * i for i in self.dilation_rate]
        self.Conv_a = tf.keras.layers.Conv3D(self.nf, self.kernel_size, strides=1, padding='valid', use_bias=True,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer='zeros')
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
        input_a = pre_padding(input, pad_size=self.padding_size_a, axis=[1, 2, 3], mode="REFLECT")
        # tf.print(input_a.shape)
        input_b = pre_padding(input, pad_size=self.padding_size_b, axis=[1, 2, 3], mode="REFLECT")
        # tf.print(input_b.shape)
        if self.lastLayer:
            u_a = self.Conv_a(input_a)
            u_b = self.Conv_b(input_b)
        else:
            u_a = self.ReLU(self.Conv_a(input_a))
            # tf.print(u_a.shape)
            u_b = self.ReLU(self.Conv_b(input_b))
            # tf.print(u_b.shape)
        u = u_a + u_b
        a = self.Softmax(self.W_a(self.AvgPool(u)))  # no hidden layer
        # a = self.Softmax(self.W_a(self.W_r(self.AvgPool(u))))  # hidden layer with r=2

        # # print params
        # tf.print("u_a", u_a.shape, u_a.dtype)
        # tf.print("u_b", u_b.shape, u_b.dtype)
        # tf.print("s", self.AvgPool(u).shape, self.AvgPool(u).dtype)
        # tf.print("a", a.shape, a.dtype, a)
        # scio.savemat('./sk_a' + str(self.layer_no) + '.mat', {'sk_a': a.numpy()})
        # tf.print("b", 1-a)
        # scio.savemat('./sk_b' + str(self.layer_no) + '.mat', {'sk_b': (1-a).numpy()})

        if self.lastLayer:
            return tf.cast(u_a * a + u_b * (1-a), dtype='float32')
        return u_a * a + u_b * (1-a)


class Nw_SK(tf.keras.layers.Layer):
    """
    Nw block using Selective Kernel Networks i.e. SKNet
    """
    def __init__(self, n_l=5, n_f=32, kernel_initializer=tf.keras.initializers.GlorotNormal()):
        super(Nw_SK, self).__init__()
        self.n_l = n_l
        self.n_f_list = [n_f] * (n_l - 1) + [2]
        # self.n_l = 6
        # self.n_f_list = [18, 36, 64, 36, 18, 2]
        self.kernel_initializer = kernel_initializer
        self.cell_list = []
        for l in range(self.n_l):
            if l == self.n_l - 1:
                self.cell_list.append(SK_Layer(2, self.kernel_initializer, lastLayer=True, layer_no=self.n_l))
            else:
                self.cell_list.append(SK_Layer(self.n_f_list[l], self.kernel_initializer, lastLayer=False, layer_no=l + 1))

    def call(self, input, training=True):
        nw = {'c_' + str(0): input}
        for l in range(0, self.n_l):
            # tf.print('c_' + str(l), nw['c_' + str(l)].shape)
            # tf.print('c_' + str(l), nw['c_' + str(l)].dtype)
            nw['c_' + str(l + 1)] = self.cell_list[l](nw['c_' + str(l)], training=training)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].shape)
        # tf.print('c_' + str(l+1), nw['c_' + str(l + 1)].dtype)
        return nw['c_' + str(l + 1)]
