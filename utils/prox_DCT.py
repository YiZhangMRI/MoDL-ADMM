import tensorflow as tf
import numpy as np
from utils.operator import complex2c, pre_padding
from utils.denoiser_FoE import sphere_projection
import scipy.io as scio


class DCT_2D_filter(tf.keras.layers.Layer):
    """
    2D block-wise DCT filtering for image smoothing (no grad)
    """

    def __init__(self, radius=3.0, trainable=True):
        super(DCT_2D_filter, self).__init__()
        self.radius = radius
        self.trainable = trainable
        self.nf = self.radius ** 2 - 1

    def build(self, input_shape):
        filter = self.make_DCT_basis()  # return DCT basis as numpy array
        init_kernel = np.expand_dims(filter[:, :, 1:self.nf + 1], axis=(0, 3))
        # [1, r, r, 1, (r**2-1)] [filter_depth, filter_height, filter_width, in_channels, out_channels]
        self.kernel_initializer = tf.keras.initializers.constant(init_kernel)
        self.DCT_Conv = tf.keras.layers.Conv3D(self.nf, (1, self.radius, self.radius),
                                               strides=1, padding='valid', use_bias=False,
                                               kernel_initializer=self.kernel_initializer,
                                               kernel_constraint=lambda x: sphere_projection(x, norm_bound=1.0, axis=(1,2)),
                                               trainable=self.trainable)

    def make_DCT_basis(self):
        DCT_basis = np.zeros(shape=[self.radius, self.radius, self.radius ** 2])
        XX, YY = np.meshgrid(range(self.radius), range(self.radius))
        C = np.ones(self.radius)
        C[0] = 1 / np.sqrt(2)
        for v in range(self.radius):
            for u in range(self.radius):
                x_rad = (2 * XX + 1) * u * np.pi / (2 * self.radius)
                y_rad = (2 * YY + 1) * v * np.pi / (2 * self.radius)
                DCT_basis[:, :, u + v * self.radius] = (2 * C[v] * C[u] / self.radius) * np.cos(x_rad) * np.cos(y_rad)
        return DCT_basis.astype(np.float32)

    def call(self, x):
        x = complex2c(x, inv=False)  # [nb, nz, nx, ny, 2]
        padded_x = pre_padding(x, pad_size=[self.radius // 2, self.radius // 2], axis=[2, 3], mode="REFLECT")
        feature_real = self.DCT_Conv(tf.expand_dims(padded_x[:, :, :, :, 0], axis=-1))  # [nb, nz, nx, ny, nf]
        feature_imag = self.DCT_Conv(tf.expand_dims(padded_x[:, :, :, :, 1], axis=-1))  # [nb, nz, nx, ny, nf]
        # extract DCT_x for each basis
        DCT_x = tf.expand_dims(complex2c(tf.concat([feature_real[:, :, :, :, 0:1],
                                                    feature_imag[:, :, :, :, 0:1]], axis=-1),
                                         inv=True), axis=-1)
        for f in range(1, self.nf):
            DCT_x = tf.concat([DCT_x, tf.expand_dims(complex2c(tf.concat([feature_real[:, :, :, :, f:f+1],
                                                                          feature_imag[:, :, :, :, f:f+1]], axis=-1),
                                                               inv=True), axis=-1)], axis=-1)

        return DCT_x


def softThres(x, thres):
    """
    Soft threshold: sign(x)*max(|x|-thres, 0)
    """
    x_abs = tf.abs(x)
    coef = tf.nn.relu(x_abs - thres) / (x_abs + 1e-10)
    coef = tf.cast(coef, tf.complex64)
    return coef * x


def prox_DCTxy(x, DCT_x, rho=1, miu_list=[0.1] * 8):
    """
    Approximation to Prox_DCT_rho/miu(x) at x-y plane
    """
    n_f = DCT_x.shape[-1]  # num of regular terms
    thres_list = tf.cast(miu_list / rho, tf.complex64)
    proxDCT = tf.zeros_like(x)
    for f in range(n_f):
        DCT = DCT_x[:, :, :, :, f]
        ST_DCT = softThres(DCT, tf.abs(thres_list[f]))
        delta = thres_list[f] * (ST_DCT - DCT)
        proxDCT += (x * (1 + thres_list[f]) + delta) / (1 + thres_list[f] ** 2)
    return proxDCT / n_f
