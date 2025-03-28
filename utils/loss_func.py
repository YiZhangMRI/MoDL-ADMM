import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.prox_TV import TV_x, TV_y


class compute_Loss():

    def __init__(self, mask, ratio=10.0, stage=2):
        self.mask = mask
        self.ratio = ratio
        self.stage = stage

    def __call__(self, label, recon, batch_size):

        if self.stage == 1:
            # train for denoising, L2_img + L2_CEST
            loss_CEST = L2norm(CEST_diff(recon), CEST_diff(label))
            loss_mse = mse(recon, label)
            loss_L2 = L2norm(recon, label)
            loss_TV = TVxy(recon)
            loss_train = loss_L2 + self.ratio * loss_CEST

            loss_CEST = loss_CEST / batch_size
            loss_TV = loss_TV / batch_size
            loss_mse = loss_mse / batch_size
            loss_L2 = loss_L2 / batch_size
            loss_train = loss_train / batch_size
            return loss_mse, loss_CEST, loss_TV, loss_L2, loss_train

        elif self.stage == 2:
            loss_CEST = L2norm(CEST_diff(recon), CEST_diff(label))
            loss_mse = mse(recon, label)
            loss_L2 = L2norm(recon, label)
            loss_TV = TVxy(recon)
            loss_train = (loss_L2 + self.ratio * loss_CEST) / (self.ratio / 10 + 1) * 2

            loss_CEST = loss_CEST / batch_size
            loss_TV = loss_TV / batch_size
            loss_mse = loss_mse / batch_size
            loss_L2 = loss_L2 / batch_size
            loss_train = loss_train / batch_size
            return loss_mse, loss_CEST, loss_TV, loss_L2, loss_train


def mse(recon, label):  # mean square erro
    if recon.dtype == 'complex64':
        residual_cplx = recon - label
        residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        mse = tf.reduce_mean(residual ** 2)
    else:
        residual = recon - label
        mse = tf.reduce_mean(residual ** 2)
    return mse


def L2norm(recon, label):  # Frobenius norm
    if recon.dtype == 'complex64':
        residual = recon - label
        residual = tf.stack([tf.math.real(residual), tf.math.imag(residual)], axis=-1)
    else:
        residual = recon - label
    L2norm = tf.math.reduce_sum(tf.math.pow(residual, 2))
    return L2norm


def L1norm(recon, label):  # L1 norm
    if recon.dtype == 'complex64':
        residual = recon - label
        residual = tf.stack([tf.math.real(residual), tf.math.imag(residual)], axis=-1)
    else:
        residual = recon - label
    L1norm = tf.math.reduce_sum(tf.math.abs(residual))
    return L1norm


def CEST_diff(recon):
    # CEST contrast mapping, return img with shape [nb, nx, ny*nf/2]
    num_omega = int(recon.shape[1] / 2) - 1
    # S0 = recon[:, 0:1, :, :]
    # recon = recon / S0
    residual = recon[:, 1, :, :] - recon[:, -1, :, :]
    for omega in range(num_omega - 1):
        residual = tf.concat([residual, recon[:, omega + 2, :, :] - recon[:, -2 - omega, :, :]], axis=-1)
    residual = tf.concat([residual, recon[:, num_omega + 1, :, :]], axis=-1)  # concat with 0ppm original
    return residual


def TVxy(recon):
    TVx = TV_x(recon)
    TVx = tf.math.reduce_mean(tf.math.abs(TVx))
    # tf.print("TVx", TVx)
    TVy = TV_y(recon)
    TVy = tf.math.reduce_mean(tf.math.abs(TVy))
    # tf.print("TVy", TVy)
    return TVx + TVy
