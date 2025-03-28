import tensorflow as tf
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


def cal_DN_ratio(image, enable=True):
    """
    calculate data normalization ratio by smoothed img, i.e. the estimate avg Z-spetrum
    """
    if enable:
        full_k = fft2c_mri(image)
        DN_k = tf.zeros((full_k.shape[0], full_k.shape[1], full_k.shape[2], full_k.shape[3]),
                        dtype=tf.dtypes.complex64)
        [_, _, nx, ny] = np.shape(image)
        # get the central 12x12/6x6 k-space data
        width = 3
        DN_k_temp = tf.concat(values=[DN_k[:, :, :nx // 2 - width, :], full_k[:, :, nx // 2 - width:nx // 2 + width, :],
                                      DN_k[:, :, nx // 2 + width:, :]], axis=2)
        DN_k = tf.concat(values=[DN_k[:, :, :, :ny // 2 - width], DN_k_temp[:, :, :, ny // 2 - width:ny // 2 + width],
                                 DN_k[:, :, :, ny // 2 + width:]], axis=3)
        DN_image = ifft2c_mri(DN_k)
        # cal DS ratio only use central 40x40 region
        DN_ratio = tf.abs(
            tf.reduce_mean(DN_image[:, :, nx // 2 - 20:nx // 2 + 20, ny // 2 - 20:ny // 2 + 20], axis=[2, 3]))
        DN_ratio = tf.cast(DN_ratio / DN_ratio[:, 0:1], dtype=tf.dtypes.complex64)
    else:
        # if normalization not activated
        DN_ratio = tf.ones([image.shape[0], image.shape[1], 1], dtype=tf.dtypes.complex64)
    DN_ratio = tf.reshape(DN_ratio, [image.shape[0], 1, image.shape[1], 1, 1])
    return DN_ratio


def pre_padding(x, pad_size=[3, 11, 11], axis=[1, 2, 3], mode="REFLECT"):
    """
    <pre_padding>: since we are using huge kernel size 11x11, each frames should be pre-padded using 'reflect' mode,
                   then cut off redundant part after convolution.
    <x>: features need to be padded
    <pad_size>: list of padding size, length should be same as <axis>
    <axis>: list of axis which are padded
    <mode>: "CONSTANT", "REFLECT", or "SYMMETRIC"
    """
    paddings = []
    j = 0
    for i in range(len(x.shape)):
        if i in axis:
            paddings.append([pad_size[j], pad_size[j]])
            j = j + 1
        else:
            paddings.append([0, 0])
    x_padded = tf.pad(x, paddings=paddings, mode=mode)
    return x_padded


class Gaussian_2D_filter(tf.keras.layers.Layer):
    """
    2D gaussian filtering for image smoothing (no grad)
    """

    def __init__(self, sigma=0.5, radius=1.0, dtype=tf.float32):
        super(Gaussian_2D_filter, self).__init__()
        self.radius = tf.cast(radius, dtype=tf.int32)
        xx = tf.cast(tf.range(-self.radius, self.radius + 1), dtype=dtype)
        g = tf.exp(-0.5 * tf.square(xx / sigma))
        g = g / tf.reduce_sum(g)
        filter = tf.Variable(tf.expand_dims(g, 1) * g, trainable=False, name='gaussian_filter')
        filter_real = tf.stack(values=[filter, tf.zeros_like(filter)], axis=-1)  # filter for real channel
        filter_imag = tf.stack(values=[tf.zeros_like(filter), filter], axis=-1)  # filter for imag channel
        self.kernel = tf.stack(values=[filter_real, filter_imag], axis=-1)[tf.newaxis, :]  # add depth axis as 1
        # [1, 2*r+1, 2*r+1, 2, 2]
        # [filter_depth, filter_height, filter_width, in_channels, out_channels]

    def call(self, x):
        padded_x = pre_padding(x, pad_size=[self.radius, self.radius], axis=[2, 3], mode="REFLECT")  # pad on edges
        filtered_x = tf.nn.conv3d(padded_x, self.kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
        return tf.cast(filtered_x, dtype=tf.float32)


def softthres(x, thres):
    """
    soft threshold: sign(x)*max(|x|-thres, 0)
    """
    x_abs = tf.abs(x)
    coef = tf.nn.relu(x_abs - thres) / (x_abs + 1e-10)
    coef = tf.cast(coef, tf.complex64)
    return coef * x


class data_share_block(tf.keras.layers.Layer):
    """
    under sample k-space data and share neighboring data
    input:
    mask: nd array [nt,nx,ny] (e.g. [54,96,96])
    k: nd array, k-space data [nb,nc,nt,nx,ny] (e.g. [1,16,54,96,96])
    output:
    k_share: nd array, under sampled k-space data [nb,nc,nt,nx,ny] (e.g. [1,16,54,96,96])
    """

    def __init__(self, mask, alpha=1.0, beta=1.0):
        super(data_share_block, self).__init__()
        self.mask = mask  # [54,96,96]
        mask_temp = tf.zeros(shape=[self.mask.shape[0] + 2, self.mask.shape[1], self.mask.shape[2]],
                             dtype=np.complex64)
        mask_temp = tf.concat(values=[mask_temp[:1, :, :], self.mask, mask_temp[-1:, :, :]], axis=0)
        # # S0 do not share data since different to others
        # mask_temp = tf.concat(values=[mask_temp[:2, :, :], self.mask[1:, :, :], mask_temp[-1:, :, :]], axis=0)
        mask_temp = tf.reshape(mask_temp, shape=(1, 1) + mask_temp.shape)
        self.mask_bool = tf.cast(mask_temp, dtype='bool')  # [1,1,56,96,96]
        # how many k-space to share about, default to 100%
        self.alpha = tf.constant(alpha, dtype=tf.complex64)
        self.beta = tf.constant(beta, dtype=tf.complex64)

    def call(self, k):
        k_temp = tf.zeros((k.shape[0], k.shape[1], k.shape[2] + 2, k.shape[3], k.shape[4]),
                          dtype=np.complex64)
        k_temp = tf.concat(values=[k_temp[:, :, :1, :, :], k, k_temp[:, :, -1:, :, :]], axis=2)  # [1,16,56,96,96]
        k_share = tf.zeros(k.shape, dtype=np.complex64)  # [1,16,54,96,96]
        for i in range(1, k_share.shape[2] + 1):  # frame 1 to 54
            mask_2 = self.mask_bool[:, :, i, :, :]
            mask2 = (~self.mask_bool[:, :, i, :, :])
            mask1 = self.mask_bool[:, :, i - 1, :, :]
            mask_1 = tf.logical_and(mask2, mask1)
            # find samples in frame_(i-1) but not in frame_i
            mask_tmp = tf.logical_or(self.mask_bool[:, :, i - 1, :, :], self.mask_bool[:, :, i, :, :])
            mask_tmp = ~mask_tmp
            mask_3 = tf.logical_and(self.mask_bool[:, :, i + 1, :, :], mask_tmp)
            # find samples not in frame_(i-1) and frame_i, but in frame_(i+1)
            mask_1 = tf.cast(mask_1, dtype=np.complex64)  # samples to share in frame_(i-1)
            mask_2 = tf.cast(mask_2, dtype=np.complex64)  # samples to share in frame_(i)
            mask_3 = tf.cast(mask_3, dtype=np.complex64)  # samples to share in frame_(i+1)
            k_layer = mask_1 * k_temp[:, :, i - 1:i, :, :] * self.alpha + \
                      mask_2 * k_temp[:, :, i:i + 1, :, :] + \
                      mask_3 * k_temp[:, :, i + 1:i + 2, :, :] * self.beta
            k_share = tf.concat(values=[k_share[:, :, :i - 1, :, :], k_layer, k_share[:, :, i:, :, :]], axis=2)
        # # S0 do not share data since different to others
        # k_share = tf.concat(values=[k[:, :, 0:1, :, :] * self.mask[0:1, :, :],
        #                             k_share[:, :, 1:, :, :]], axis=2)
        return k_share


def complex2c(into, inv=False):
    """
    transfer between complex64 and pseudo-complex fp32
    """
    if inv:  # fp16/32 to complex64
        if len(into.shape) == 5:  # [nb, nf, nx, ny, 2]
            output = tf.complex(tf.cast(into[:, :, :, :, 0], dtype='float32'),
                                tf.cast(into[:, :, :, :, 1], dtype='float32'))
        else:  # [nb, nc, nf, nx, ny, 2]
            output = tf.complex(tf.cast(into[:, :, :, :, :, 0], dtype='float32'),
                                tf.cast(into[:, :, :, :, :, 1], dtype='float32'))
    else:  # complex64 to fp32
        if len(into.shape) == 4:  # [nb, nf, nx, ny]
            output = tf.stack([tf.math.real(into), tf.math.imag(into)], axis=-1)
        else:  # [nb, nf, nx, ny, 1]
            output = tf.concat([tf.math.real(into), tf.math.imag(into)], axis=-1)
    return output


class data_recon_block(tf.keras.layers.Layer):
    """
    data re-consistence layer by projection
    <k_rec> is rearranged with original k-space data <k_org> according to <mask>
    """

    def __init__(self, mask):
        super(data_recon_block, self).__init__()
        self.mask = mask

    def call(self, x_rec, k_org, csm):
        if len(k_org.shape) == 4 or csm is None:  # single coil
            k_rec = fft2c_mri(x_rec)
            k_rec = (1 - self.mask) * k_rec + self.mask * k_org
            x_rec = ifft2c_mri(k_rec)
        else:  # multi coil
            k_rec = fft2c_mri(tf.expand_dims(x_rec, 1) * csm)
            k_rec = (1 - self.mask) * k_rec + self.mask * k_org
            x_rec = ifft2c_mri(k_rec) * tf.math.conj(csm)
            x_rec = tf.reduce_sum(x_rec, 1)  # / tf.cast(tf.reduce_sum(tf.abs(csm) ** 2, 1), dtype=tf.complex64)
        return x_rec


def noisy_sample(k, mask, sigma=0.0, ratio=1.0):
    """
    retrospective sampling with additive white noise
    """
    if sigma > 0.0:
        noise = tf.complex(tf.random.normal(k.shape, mean=0.0, stddev=sigma / 1.414),
                           tf.random.normal(k.shape, mean=0.0, stddev=sigma / 1.414))
        noise = tf.where(tf.math.real(mask) > 0., noise, tf.complex(0.0, 0.0))
        noise = noise * (1 - ratio) ** (4 * ratio)
        return k * mask + noise
    else:
        return k * mask


def sos(x):
    """
    sum of squares for coil combination
    """
    # x: nb, ncoil, nt, nx, ny; complex64
    x = tf.math.reduce_sum(tf.abs(x ** 2), axis=1)
    x = x ** (1.0 / 2)
    return x


def fft2c_mri(x):
    """
    input image x: [nb, nc, nt, nx, ny] for multi-coil or [nb, nt, nx, ny] for single-coil
    """
    nx = np.float32(x.shape)[-2]
    ny = np.float32(x.shape)[-1]
    x = tf.signal.fftshift(x, -2)
    x = tf.signal.fftshift(x, -1)
    X = tf.signal.fft2d(x)
    X = tf.signal.fftshift(X, -1) / tf.sqrt(np.complex64(ny + 0j))
    X = tf.signal.fftshift(X, -2) / tf.sqrt(np.complex64(nx + 0j))
    return X


def ifft2c_mri(X):
    """
    input ksp X: [nb, nc, nt, nx, ny] for multi-coil or [nb, nt, nx, ny] for single-coil
    """
    nx = np.float32(X.shape)[-2]
    ny = np.float32(X.shape)[-1]
    X = tf.signal.fftshift(X, -2)
    X = tf.signal.fftshift(X, -1)
    x = tf.signal.ifft2d(X)
    x = tf.signal.fftshift(x, -1) * tf.sqrt(np.complex64(ny + 0j))
    x = tf.signal.fftshift(x, -2) * tf.sqrt(np.complex64(nx + 0j))
    return x


class Amat():
    """
    sub-sample Fourier transform operator A=S*F*C
    where S stands for sampling, F for FT, C for coil sensitivity
    input:
    multi/single-coil ksp or img <b>
    coil sensitivity map <csm>
    <inv>:
    set True for backward transformation img -> masked ksp
    set False for forward transformation ksp -> aliasing img
    """

    def __init__(self, mask):
        super(Amat, self).__init__()
        self.mask = mask

    def mtimes(self, b, csm, inv):
        if csm is None:  # single coil
            if inv:
                x = ifft2c_mri(b * self.mask)
            else:
                x = fft2c_mri(b) * self.mask
        else:  # multi coil
            if inv:
                # This is the 'sense encoding' theory, but only used for iteration methods which has inverse calculation
                x = ifft2c_mri(b * self.mask)  # broadcast
                x = x * tf.math.conj(csm)
                x = tf.reduce_sum(x, 1)  # / tf.cast(tf.reduce_sum(tf.abs(csm) ** 2, 1), dtype=tf.complex64)
            else:
                b = tf.expand_dims(b, 1) * csm
                x = fft2c_mri(b) * self.mask
        return x
