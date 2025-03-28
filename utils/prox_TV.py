import tensorflow as tf


def TV_x(x):
    """
    Total variation along x-axis
    """
    y = tf.concat(values=[tf.expand_dims(x[:, :, -1, :], axis=2), x[:, :, :-1, :]], axis=2)
    return x - y


def TV_xT(x):
    """
    Transposed total variation along x-axis
    """
    y = tf.concat(values=[x[:, :, 1:, :], tf.expand_dims(x[:, :, 0, :], axis=2)], axis=2)
    return x - y


def TV_star_x(TVx):
    """
    The adjugate matrix of TV_x, integrate along x-axis
    """
    nx = TVx.shape[2]
    TVstarx = tf.expand_dims(TVx[:, :, 0, :], axis=2)
    for i in range(1, nx):
        new_row = tf.expand_dims(TVx[:, :, i, :] + TVstarx[:, :, -1, :], axis=2)
        TVstarx = tf.concat([TVstarx, new_row], axis=2)
    return TVstarx * -1


def TV_y(x):
    """
    Total variation along y-axis
    """
    y = tf.concat(values=[tf.expand_dims(x[:, :, :, -1], axis=3), x[:, :, :, :-1]], axis=3)
    return x - y


def TV_yT(x):
    """
    Transposed total variation along y-axis
    """
    y = tf.concat(values=[x[:, :, :, 1:], tf.expand_dims(x[:, :, :, 0], axis=3)], axis=3)
    return x - y


def softThres(x, thres):
    """
    Soft threshold: sign(x)*max(|x|-thres, 0)
    """
    x_abs = tf.abs(x)
    coef = tf.nn.relu(x_abs - thres) / (x_abs + 1e-10)
    coef = tf.cast(coef, tf.complex64)
    return coef * x


def prox_TVxy(x, rho=1, miu_x=0.1, miu_y=0.1):
    """
    Approximation to Prox_TV_rho/miu(x) at x-y plane
    """
    thres_x = miu_x / rho
    TVx = TV_x(x)
    STTVx = softThres(TVx, thres_x)
    TVxT = TV_xT(x)
    STTVxT = softThres(TVxT, thres_x)
    thres_y = miu_y / rho
    TVy = TV_y(x)
    STTVy = softThres(TVy, thres_y)
    TVyT = TV_yT(x)
    STTVyT = softThres(TVyT, thres_y)
    thres_x = tf.cast(thres_x, tf.complex64)
    thres_y = tf.cast(thres_y, tf.complex64)
    delta_x = ((STTVx - TVx) + (STTVxT - TVxT)) * thres_x / 2
    delta_y = ((STTVy - TVy) + (STTVyT - TVyT)) * thres_y / 2
    proxTVx = ((x * (1+thres_x) + delta_x) / (1+thres_x**2) + (x * (1+thres_y) + delta_y) / (1+thres_y**2)) / 2
    return proxTVx




