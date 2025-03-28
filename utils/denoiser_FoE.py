import tensorflow as tf
from utils.operator import pre_padding


def sphere_projection(x, zero_mean=True, l2_normalize=True, norm_bound=1.0, axis=(0, 1, 2, 3)):
    """
    <sphere_projection>: to maintain the constrain that all kernels should be 0-mean and 1-L2norm, project
                        k_real & k_image onto unit spheres after back propagation and updating
    <x>: tf variable which should be projected
    <zero_mean>: boolean True for zero-mean. default=True
    <l2_normalize>: boolean True for l_2-norm ball projection. default:True
    <norm_bound>: defines the size of the norm ball
    <axis>: defines the axis for the reduction (mean and norm) meaning all axis except out_channels
    <return>: projected <x>
    """

    # print('x', x.shape)
    # depth x width x height x in_channel x out_channel
    if zero_mean:
        x_mean = tf.math.reduce_mean(x, axis=axis, keepdims=True)
        # print('x_mean', x_mean.shape, x_mean)
        x = x - x_mean

    if l2_normalize:
        magnitude = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x), axis=axis, keepdims=True))
        # print('magnitude', magnitude.shape, magnitude)
        x = x / magnitude * norm_bound

    return x


class rbf_Act(tf.keras.layers.Layer):
    """
    RBF custom activation func with trainable weights <w> and Gaussian base
    input feature <x>: nb x nf x nx x ny x nch (x1)
    trainable weight <w>: 1 x nch x nw
    <miu>: center of RBFs 1 x nw
    <sigma>: std of RBFs 1
    """

    def __init__(self, n_w=31, n_f=24, v_min=-1.0, v_max=1.0, init_type='inv-student-t', init_scale=1.0):
        super(rbf_Act, self).__init__()
        self.n_w = n_w
        self.init_type = init_type
        self.miu = tf.linspace(v_min, v_max, n_w)
        self.sigma = 2 * v_max / (n_w - 1)
        if self.init_type == 'linear':
            w_0 = init_scale * self.miu
        elif self.init_type == 'tv':
            w_0 = init_scale * tf.math.sign(self.miu)
        elif self.init_type == 'relu':
            w_0 = init_scale * tf.math.maximum(self.miu, 0.0)
        elif self.init_type == 'student-t':
            # alpha = 100.0
            # w_0 = init_scale * tf.math.sqrt(alpha) * self.miu / (1.0 + 0.5 * alpha * self.miu ** 2)
            n = 4.0  # rank
            w_0 = init_scale * 1 / tf.math.sqrt(n) * tf.math.pow((1 + tf.math.square(self.miu) / 2 * n), -(n + 1) / 2)
        elif self.init_type == 'inv-student-t':
            # alpha = 100.0
            # w_0 = init_scale * tf.math.sqrt(alpha) * self.miu / (1.0 + 0.5 * alpha * self.miu ** 2)
            n = 4.0  # rank
            w_0 = init_scale * (
                    0.5 - 1 / tf.math.sqrt(n) * tf.math.pow((1 + tf.math.square(self.miu) / 2 * n), -(n + 1) / 2))
        else:
            raise ValueError("init_type '%s' not defined!" % init_type)
        w_0 = tf.reshape(w_0, shape=(1, 1, n_w))
        w_0 = tf.tile(tf.reshape(w_0, shape=(1, 1, n_w)), [1, n_f, 1])  # 1 x nch x nw
        self.w = tf.Variable(w_0, trainable=True, name='w', dtype=tf.float32)
        # fp16
        self.miu = tf.cast(self.miu, dtype=tf.float16)
        self.sigma = tf.cast(self.sigma, dtype=tf.float16)

    def call(self, x):
        # x = tf.cast(x, dtype=tf.float32)
        output = tf.zeros_like(x)
        w = tf.cast(self.w, dtype=tf.float16)
        # print('x', x.dtype)
        # print('miu', self.miu.dtype)
        # print('sigma', self.sigma.dtype)
        # print('w', w.dtype)
        for i in range(self.n_w):
            output = output + w[..., i] * tf.math.exp(-tf.math.square(x - self.miu[i]) / (2 * self.sigma ** 2))
            # Φ(z) = Σ w_i * exp(- (z-miu_i)^2 / 2*sigma^2)
        return output


class FoE_Layer(tf.keras.layers.Layer):
    """
    FoE modeled layer
    Convolution: filters are of size 11x11x3(x2) and 24 pairs of real & image (SK-net structure should be considered)
    RBF activation: 24 sets of 31 RBFs functions
    """

    def __init__(self, n_f=24, kernel_size=11, kernel_initializer=tf.keras.initializers.GlorotNormal(), act='RBF',
                 n_w=31, v_min=-1.0, v_max=1.0, init_type='inv-student-t', init_scale=1.0, lastLayer=False, layer_no=1):
        super(FoE_Layer, self).__init__()
        self.n_f = n_f
        self.lastLayer = lastLayer
        self.layer_no = layer_no
        self.kernel_size = (3, kernel_size, kernel_size)  # depth, width, height (in_channels, out_channels)
        self.pad_size = tf.cast(tf.math.floor([i / 2 for i in self.kernel_size]), dtype=tf.int32)
        self.kernel_initializer = kernel_initializer
        self.act = act
        if self.act == 'RBF':
            self.RBFs_Activation = rbf_Act(n_w=n_w, n_f=n_f, v_min=v_min, v_max=v_max, init_type=init_type,
                                           init_scale=init_scale)
        elif self.act == 'relu':
            self.RBFs_Activation = tf.keras.layers.LeakyReLU(alpha=0.1)
        elif self.act == 'tanh':
            self.RBFs_Activation = tf.keras.layers.Activation(tf.keras.activations.tanh)
        else:
            raise ValueError("act_type '%s' not defined!" % self.act)
        # self.Conv = tf.keras.layers.Conv3D(self.n_f, self.kernel_size, strides=1, padding='valid', use_bias=False,
        #                                    kernel_initializer=self.kernel_initializer,
        #                                    kernel_constraint=lambda x: sphere_projection(x, norm_bound=1.0,
        #                                                                                  axis=(1, 2)))
        self.Conv = tf.keras.layers.Conv3D(self.n_f, self.kernel_size, strides=1, padding='valid', use_bias=True,
                                           kernel_initializer=kernel_initializer, bias_initializer='zeros')

    @staticmethod
    def normalization(x, zero_mean=True, value_bound=1.0, axis=(1, 2, 3)):
        Max = tf.math.reduce_max(x, axis=axis, keepdims=True)
        Min = tf.math.reduce_min(x, axis=axis, keepdims=True)
        # print(Max.shape)
        x_one = (x - Min) / (Max - Min)
        if zero_mean:
            x_norm = (x_one * 2.0 - 1.0) * value_bound
        else:
            x_norm = x_one * value_bound
        return x_norm

    def call(self, x, training=True):
        # 1x53x96x96x2
        x_pad = pre_padding(x, self.pad_size, axis=[1, 2, 3], mode="REFLECT")  # 1x55x106x106x2
        f_fuse = self.Conv(x_pad)  # 1x53x96x96x24
        if self.act == 'RBF':
            # feature normalization to [-1, 1] before get into RBF / plain activation func
            f_fuse = self.normalization(f_fuse, zero_mean=True, value_bound=1.0, axis=(1, 2, 3))
        f_a = self.RBFs_Activation(f_fuse)

        # # print paras
        # kernel = np.array(self.Conv.get_weights()[0])
        # print(kernel.shape)
        # scio.savemat('./kernel' + str(self.layer_no) + '.mat', {'kernel': kernel})
        # if self.act == 'RBF':
        #     weight = np.array(self.RBFs_Activation.get_weights())
        #     print(weight.shape)
        #     scio.savemat('./weight' + str(self.layer_no) + '.mat', {'weight': weight})
        # else:
        #     print('using %s activation' % self.act)
        # # print features
        # print('f_fuse', f_fuse.shape)
        # scio.savemat('./feature' + str(self.layer_no) + '.mat', {'feature': f_fuse.numpy()})
        # print('f_a', f_a.shape)
        # scio.savemat('./feature_activate' + str(self.layer_no) + '.mat', {'feature_activate': f_a.numpy()})
        # # quit()
        return f_a


class Nw_FoE(tf.keras.layers.Layer):
    """
    Nw block using FoE model
    input single_coil frames with real & image channel <x>: nb x nf x nx x ny x 2
    """

    def __init__(self, n_l=2, n_f=24, kernel_size=11, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                 n_w=31, v_min=-1.0, v_max=1.0, init_type='inv-student-t', init_scale=1.0):
        super(Nw_FoE, self).__init__()
        self.n_l = n_l - 1  # last layer is 2 channel plain conv3D
        self.cell_list = []
        # # full RBF
        # for l in range(self.n_l):
        #     self.cell_list.append(FoE_Layer(n_f=n_f, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
        #                                     n_w=n_w, v_min=v_min, v_max=v_max, init_type=init_type,
        #                                     init_scale=init_scale, lastLayer=False, layer_no=l + 1))

        # plain activation + RBF in last layer
        for l in range(self.n_l - 1):
            self.cell_list.append(FoE_Layer(n_f=n_f, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                            n_w=n_w, v_min=v_min, v_max=v_max, init_type=init_type,
                                            init_scale=init_scale, lastLayer=False, layer_no=l + 1, act='relu'))
        self.cell_list.append(FoE_Layer(n_f=n_f, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                        n_w=n_w, v_min=v_min, v_max=v_max, init_type=init_type,
                                        init_scale=init_scale, lastLayer=False, layer_no=self.n_l, act='RBF'))

        # 2ch 3x3x3 conv3D, similar to Nw_CNN
        self.cell_list.append(tf.keras.layers.Conv3D(2, 3, strides=1, padding='same', use_bias=True,
                                                     kernel_initializer=kernel_initializer, bias_initializer='zeros'))

        # # 2ch 1x1 conv2D, meaning weighted channel-wise sum
        # self.cell_list.append(tf.keras.layers.Conv2D(2, 1, strides=1, padding='same', use_bias=False,
        #                                              kernel_initializer=kernel_initializer))

    def call(self, x, training=True):
        nw = {'x_0': x}
        for l in range(0, self.n_l + 1):
            # tf.print('x_' + str(l), nw['x_' + str(l)].shape)
            # tf.print('x_' + str(l), nw['x_' + str(l)].dtype)
            nw['x_' + str(l + 1)] = self.cell_list[l](nw['x_' + str(l)], training=training)
        # tf.print('x_' + str(l+1), nw['x_' + str(l + 1)].shape)
        # tf.print('x_' + str(l+1), nw['x_' + str(l + 1)].dtype)
        return tf.cast(nw['x_' + str(l + 1)], dtype=tf.float32)
