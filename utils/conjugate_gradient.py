import tensorflow as tf
from utils.operator import Amat, complex2c
import scipy.io as scio


class LHS_inv:
    """
    (left hand side)^-1, (A^HA+λI)*img
    """

    def __init__(self, mask, lam):
        self.A = Amat(mask)
        self.lam = tf.complex(lam, 0.0)

    def ops(self, img, csm):
        AHA = self.A.mtimes(self.A.mtimes(img, csm, False), csm, True)
        return AHA + self.lam * img


class DC(tf.keras.layers.Layer):
    """
    DC block calculating x=(LHS)*(RHS)
    LHS = (LHS_inv)^-1 = (A^HA+λI)^-1
    RHS = Dw = (A^Hb+lam*z)
    """

    def __init__(self, mask, gradientMethod='AG', K=10):
        super(DC, self).__init__()
        self.K = K
        self.mask = mask
        self.gradientMethod = gradientMethod

    def call(self, RHS, CSM, X_est, training=True, lam=0.01):
        def fn(elem):
            rhs, csm, x_est = elem
            # recover the batch axis after fn_map
            rhs = tf.expand_dims(rhs, axis=0)
            csm = tf.expand_dims(csm, axis=0)
            x_est = tf.expand_dims(x_est, axis=0)
            if self.gradientMethod == 'AG':
                CG = CG_block(self.lhs_i, csm, self.K)
                return tf.squeeze(CG(rhs, x_est), axis=0)
            elif self.gradientMethod == 'MG':
                if training:
                    return tf.squeeze(self.CG_MG(rhs, csm, x_est), axis=0)
                else:
                    CG = CG_block(self.lhs_i, csm, self.K)
                    return tf.squeeze(CG(rhs, x_est), axis=0)
        elem = (RHS, CSM, X_est)
        self.lhs_i = LHS_inv(mask=self.mask, lam=lam)  # initialize LHS_inv operator
        X = tf.map_fn(fn, elem, dtype=tf.complex64)
        return X

    @tf.custom_gradient
    def CG_MG(self, RHS, csm, x_est):
        # y = (A^HA + λI) ^ -1 * RHS, RHS = A^Tb + λz_k
        CG = CG_block(self.lhs_i, csm, self.K)
        def grad(upstream):
            # ▽z_k-1_C=(A^HA+λI)^-1*▽x_k_C
            # tf.print('upstream', upstream.shape, upstream.dtype)
            # scio.savemat('./upstream.mat', {'upstream': upstream.numpy()})
            downstream = CG(upstream, tf.zeros_like(upstream))
            # tf.print('downstream', downstream.shape, downstream.dtype)
            # scio.savemat('./downstream.mat', {'downstream': downstream.numpy()})
            return downstream, tf.zeros_like(csm), tf.zeros_like(x_est)  # zero grad for constant args csm & x_est
        return CG(RHS, x_est), grad


class CG_block(tf.keras.layers.Layer):
    """
    input LHS_inv = (A^HA+λI), and RHS = (A^Hb+lam*z)
    then calculate x = LHS * RHS using conjugate gradient method
    question equal to solving RHS = LHS_inv * x (use <Q> instead of LHS_inv for simplicity)
    <r> is the residual distance, <p> is the conjugate gradient vector:
        if set x_0 = 0, then r_0 = p_0 = RHS - LHS_inv * x_0 = RHS
        elif set x_0 = x_est, r_0 = p_0 = RHS - LHS_inv * x_est
        r_{k+1} = r_k - Qp_k
    <alpha> is learning rate of every step p_k:
        alpha = rTr_k / pTQp_k
        x_{k+1} = x_k-1 + alpha * p_k
    <beta> is the momentum of last gradient p_k:
        (notice: p_{k+1} should be r_{k+1} -Σλ_i*p_i, but simplified to momentum update)
        beta = rTr_{k+1} / rTr_k
        p_{k+1} = r_{k+1} + beta * p_k
    """

    def __init__(self, LHS_inv, csm, K=10):
        super(CG_block, self).__init__()
        self.LHS_inv = LHS_inv
        self.csm = csm
        self.K = K
        self.cond = lambda i, rTr, *_: tf.logical_and(tf.less(i, self.K), rTr > 1e-8)

    def call(self, RHS, x_est):
        # while step<self.K and rTr>1e-6
        def update(i, rTr, x, r, p):
            # LHS^-1 is simplified as <Q>
            Qp = self.LHS_inv.ops(p, self.csm)
            # learning rate <alpha>
            alpha = rTr / tf.cast(tf.math.reduce_sum(tf.math.conj(p) * Qp), dtype='float32')
            # tf.print('CG iter %d, res rTr %.3e, lr alpha %.3e' % (i, rTr, alpha))
            alpha = tf.complex(alpha, 0.0)
            x = x + alpha * p  # update <x> with grad decent
            # scio.savemat('./cg_'+str(i+1)+'.mat', {'cg_'+str(i+1): x.numpy()})
            r = r - alpha * Qp  # update residual <r>
            rTrNew = tf.cast(tf.math.reduce_sum(tf.math.conj(r) * r), dtype='float32')
            beta = rTrNew / rTr  # gradient momentum <beta>
            beta = tf.complex(beta, 0.0)
            p = r + beta * p  # update <p> with new residual <r> and history grad <beta * p>
            return i+1, rTrNew, x, r, p
        # x = tf.zeros_like(RHS)  # set x_0 = 0
        # i, r, p = 0, RHS, RHS
        x = x_est  # set x_0 = k-space_share(b)
        r = RHS - self.LHS_inv.ops(x, self.csm)
        i, p = 0, r
        rTr = tf.cast(tf.math.reduce_sum(tf.math.conj(r) * r), dtype='float32')
        loopVar = i, rTr, x, r, p
        out = tf.while_loop(self.cond, update, loopVar, parallel_iterations=1)[2]
        return out
