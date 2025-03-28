import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.operator import complex2c, data_share_block, data_recon_block, ifft2c_mri, Amat
from utils.denoiser_CNN import Nw_CNN
from utils.denoiser_SKnet import Nw_SK
from utils.conjugate_gradient import DC
from utils.prox_TV import prox_TVxy

DEFAULT_OPTS = {
    'niter': 10,
    'share_weight': 'none',
    'copy_Nw': False,
    'denoiser': 'SK',
    'nLayer': 5,
    'N0_fully_sampled': False,
    'gradMethod': 'AG',
    'CG_K': 10,
    'att_norm': False
}


class MoDL_Net_ADMM_TV(tf.keras.Model):
    """
    MoDL-ADMM model using CNN denoiser (SKnet) and CG DC-block,
    several modification were made on original MoDL, including TV regularization and ADMM unrolling
    argmin_x {|Ax-b|^2 + lam*|Nw|^2 + miu*|TVx|_1}
    """

    def __init__(self, mask, **kwargs):
        super(MoDL_Net_ADMM_TV, self).__init__(name='MoDL_Net_ADMM')
        self.options = DEFAULT_OPTS
        for key in self.options.keys():
            if key in kwargs:
                self.options[key] = kwargs[key]
        self.mask = mask  # sampling mask
        self.niter = self.options['niter']  # num of iterations
        self.share_weight = self.options['share_weight'].lower()  # whether to share learnable weights or not
        self.N0_fully_sampled = self.options['N0_fully_sampled']  # is inf ppm fully sampled
        self.data_share_block = data_share_block(self.mask)  # ksp data sharing for img estimation
        self.data_recon_block = data_recon_block(self.mask)  # data consistency by ksp subspace projection
        # data consistency by CG decent
        self.DC = DC(mask=self.mask, gradientMethod=self.options['gradMethod'], K=self.options['CG_K'])
        self.lam = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True,
                               constraint=tf.keras.constraints.NonNeg(), name='lambda')  # regularization weight
        if self.share_weight in ('nw', 'none'):
            self.lam_list = tf.Variable(tf.constant([0.01] * 10, dtype=tf.float32), trainable=True,
                                        constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-4, clip_value_max=0.1),
                                        name='lambda_list')
        self.Nw = self.create_Nw()  # regular net
        self.Nw_list = {}
        if self.share_weight in ('all', 'nw'):  # copy Nw to iteration blocks
            for i in range(1, self.niter + 1):
                self.Nw_list['Nw_%s' % i] = self.Nw
        elif self.share_weight == 'none':  # create new Nw for each iteration block
            self.Nw_list['Nw_%s' % 1] = self.Nw
            for i in range(2, self.niter + 1):
                self.Nw_list['Nw_%s' % i] = self.create_Nw()
        # regular weights for TV term
        self.miux_list = tf.Variable(tf.constant([0.01] * 10, dtype=tf.float32), trainable=True,
                                     constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-6, clip_value_max=0.1),
                                     name='miux_list')
        self.miuy_list = tf.Variable(tf.constant([0.01] * 10, dtype=tf.float32), trainable=True,
                                     constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-6, clip_value_max=0.1),
                                     name='miuy_list')
        # Lagrange penalty weight
        self.rho_list = tf.Variable(tf.constant([0.1] * 11, dtype=tf.float32), trainable=True,
                                    constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-6, clip_value_max=1.0),
                                    name='rho_list')
        self.tmp = {}  # intermediate vars cache

    def create_Nw(self):
        if self.options['denoiser'] == 'CNN':
            return Nw_CNN(n_l=self.options['nLayer'], n_f=64, kernel_size=3,
                          kernel_initializer=tf.keras.initializers.GlorotNormal())
        elif self.options['denoiser'] == 'SK':
            return Nw_SK(n_l=self.options['nLayer'], n_f=32, kernel_initializer=tf.keras.initializers.GlorotNormal())
        else:
            raise ValueError("Unsupported Nw type")

    def build(self, input_shape):
        tf.print("model built with input_shape", input_shape)
        if self.share_weight in ('nw', 'none'):
            if self.lam_list[0] == 0.01 and self.lam != 0.01:
                tf.print("# initial copy lam weight to lam_list[0:niter]")
                # first use freely lam_list, copy <lam> trained in 'share_weight:all' to <lam_list[:]>
                self.lam_list[0:self.niter].assign([self.lam] * self.niter)
            elif self.lam_list[self.niter - 1] == 0.01 and self.niter > 1:
                # pre-trained lam_list with shorter niter, copy <lam_list[past]> to <lam_list[past+1:niter]>
                past = 0
                for i in range(0, self.niter):
                    if self.lam_list[i] != 0.01:
                        past = i
                    else:
                        tf.print("# initial copy last lam_list %d weight to lam_list %d" % (past, i))
                        self.lam_list[i].assign(self.lam_list[past])
            tf.print('lam built', self.lam)
            tf.print('lam_list built', self.lam_list.numpy())
            tf.print('miux_list built', self.miux_list.numpy())
            tf.print('miuy_list built', self.miuy_list.numpy())
            tf.print('rho_list built', self.rho_list.numpy())

            if self.share_weight == 'none' and self.niter > 1:
                # create dummy data to built up weights
                dummy_input = tf.zeros([1] + input_shape[2:] + [2])
                for i in range(1, self.niter + 1):
                    dummy_output = self.Nw_list['Nw_%s' % i](dummy_input, training=False)
                if "past" not in locals():
                    past = 0
                past = past + 1
                if self.options['copy_Nw']:
                    for i in range(past + 1, self.niter + 1):
                        # copy Nw_(past+1)'s weights to Nw_[past+2:niter] as original weights
                        tf.print("# initial copy Nw_list %d weight to Nw_list %d" % (past, i))
                        self.Nw_list['Nw_%s' % i].set_weights(self.Nw_list['Nw_%s' % past].get_weights())
                tf.print('Nw_%d built' % past, self.Nw_list['Nw_%s' % past].get_weights()[1])
                tf.print('Nw_%d built' % i, self.Nw_list['Nw_%s' % i].get_weights()[1])

    def call(self, k, csm, training=True, stage=2):
        """
        input:
            k: multi coil sub-sampled k-space [nb, nc, nf, nx, ny]
            csm: coil sensitivity map [nb, nc, 1, nx, ny]
            training: control BN working under train/infer mode and CG block record grad/or not under 'MG' mode
        If stage 1: data only pass through Nw, aiming for training a noise estimator which could recover img from
        unsampled and noisy k-space data. Namely the pre-training stage in MoDL-SToRM
        Elif stage 2: we activate whole pipeline of MoDL, the data go through both Nw CNN & CG data-consistency blocks
        """
        x_0 = tf.reduce_sum(ifft2c_mri(k) * tf.math.conj(csm), 1)

        def stage1():  # train Nw only, bypass the DC block
            self.DC.trainable = False
            self.tmp['x_0'] = complex2c(x_0, inv=False)  # 0-filled aliasing img in pseudo-complex form
            self.tmp['nw'] = self.Nw(self.tmp['x_0'], training)
            self.tmp['dw'] = self.tmp['x_0'] - self.tmp['nw']
            if self.N0_fully_sampled:  # re-concat with full sampled N0
                self.tmp['dw'] = tf.concat(values=[self.tmp['x_0'][:, 0:1, :], self.tmp['dw'][:, 1:, :]], axis=1)
            dw = complex2c(self.tmp['dw'], inv=True)
            return dw

        def stage2():  # fully functional MoDL_ADMM with stage1 pre_trained Nw
            # initial Auxiliary variable y_0 be Prox_TV(x_0)
            self.tmp['y_0'] = prox_TVxy(x_0, rho=self.rho_list[0], miu_x=self.miux_list[0], miu_y=self.miuy_list[0])
            # initial Lagrange multiplier u_0 be x_0 - y_0
            self.tmp['u_-1'] = tf.zeros_like(x_0, dtype=tf.complex64)
            self.tmp['u_0'] = x_0 - self.tmp['y_0']
            self.tmp['x_0'] = complex2c(x_0, inv=False)  # 0-filled aliasing img in pseudo-complex form
            # k-space share for a better CG start point x_est
            k_share = self.data_share_block(k)
            x_est = tf.reduce_sum(ifft2c_mri(k_share) * tf.math.conj(csm), 1)
            self.tmp['x_est'] = x_est

            for i in range(1, self.niter + 1):
                # calculate and update u_i-1
                if i > 1:
                    self.tmp['u_%s' % (i - 1)] = self.tmp['u_%s' % (i - 2)] + self.tmp['x_%s' % (i - 1)] - self.tmp[
                        'y_%s' % (i - 1)]
                    self.tmp['x_%s' % (i - 1)] = complex2c(self.tmp['x_%s' % (i - 1)], inv=False)
                self.tmp['nw_%s' % i] = self.Nw_list['Nw_%s' % i](self.tmp['x_%s' % (i - 1)], training)
                self.tmp['dw_%s' % i] = self.tmp['x_%s' % (i - 1)] - self.tmp['nw_%s' % i]
                if self.N0_fully_sampled:  # re-concat with full sampled N0
                    self.tmp['dw_%s' % i] = tf.concat(
                        values=[self.tmp['x_0'][:, 0:1, :], self.tmp['dw_%s' % i][:, 1:, :]], axis=1)

                # passing DC block using CG iteration
                delta = tf.cast(self.rho_list[i], tf.complex64) * (
                            self.tmp['y_%s' % (i - 1)] - self.tmp['u_%s' % (i - 1)])
                if self.share_weight == 'all':  # shared lam for each iter
                    rhs = complex2c(self.tmp['x_0'] + self.lam * self.tmp['dw_%s' % i], inv=True) + delta
                    self.tmp['x_%s' % i] = self.DC(rhs, csm, self.tmp['x_est'], training,
                                                   lam=self.lam + self.rho_list[i])
                else:
                    rhs = complex2c(self.tmp['x_0'] + self.lam_list[i - 1] * self.tmp['dw_%s' % i], inv=True) + delta
                    self.tmp['x_%s' % i] = self.DC(rhs, csm, self.tmp['x_est'], training,
                                                   lam=self.lam_list[i - 1] + self.rho_list[i])

                # overwrite <x_est> using <x_i> as the next CG start point
                self.tmp['x_est'] = self.tmp['x_%s' % i]

                # calculate and update y_i
                if i < self.niter:
                    self.tmp['y_%s' % i] = prox_TVxy(self.tmp['x_%s' % i], rho=self.rho_list[i],
                                                     miu_x=self.miux_list[i], miu_y=self.miuy_list[i])

                # delete vars to save RAM
                del self.tmp['nw_%s' % i], self.tmp['dw_%s' % i], self.tmp['u_%s' % (i - 2)], self.tmp['y_%s' % (i - 1)]
                if i > 1:
                    del self.tmp['x_%s' % (i - 1)]

            # finally passing dc projection Layer to overwrite k-space
            self.tmp['x_%s' % i] = self.data_recon_block(x_rec=self.tmp['x_%s' % i], k_org=k, csm=csm)

            return self.tmp['x_%s' % i]

        return tf.cond(tf.math.equal(stage, 1), stage1, stage2)
