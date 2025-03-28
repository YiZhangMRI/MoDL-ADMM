import os
import random
import tensorflow as tf
import numpy as np
import scipy.io as scio
import shutil
import math


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mask(mask_pattern, acc):
    tf.print('mask %s with acc %d' % (mask_pattern, acc))
    if mask_pattern == 'cartesian':
        mask = scio.loadmat('./mask/Mask_cartesian_54_96_96_acc_%d' % acc)['mask']
    else:
        raise ValueError("Unsupported mask pattern")
    return mask


def mycopyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        tf.print("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, os.path.join(dstpath, fname))
        tf.print("copy \"%s\" \n -> \"%s\"" % (fname, dstpath))


class Warmingup_ExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate=0.0001, warm_steps=100, decay_steps=100, decay_rate=0.95):
        self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)
        self.warm_steps = tf.constant(warm_steps, dtype=tf.float32)
        self.decay_steps = tf.constant(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.constant(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        def f1():
            return self.learning_rate * (step + 1) / self.warm_steps

        def f2():
            step_int = tf.math.floor((step - self.warm_steps) / self.decay_steps)
            return self.learning_rate * tf.math.pow(self.decay_rate, step_int)

        lr = tf.cond(tf.less(step, self.warm_steps), f1, f2)
        # tf.print('step', step, '----------------', 'lr=', lr)
        return lr


class Warmingup_CosDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate=0.0001, warm_steps=100, epochs=10, steps=100):
        self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)
        self.warm_steps = tf.constant(warm_steps, dtype=tf.float32)
        self.total_steps = tf.constant(epochs * steps, dtype=tf.float32)

    def __call__(self, step):
        def f1():
            return self.learning_rate * (step + 1) / self.warm_steps

        def f2():
            step_int = tf.math.floor((step - self.warm_steps) / self.warm_steps) * self.warm_steps
            step_int = step_int / (self.total_steps - self.warm_steps)
            angle = tf.cond(tf.less(step_int, 0.99), lambda: step_int, lambda: 0.99)
            return self.learning_rate * (tf.math.cos(angle * math.pi) * 0.5 + 0.5)

        lr = tf.cond(tf.less(step, self.warm_steps), f1, f2)
        # tf.print('step', step, '----------------', 'lr=', lr)
        return lr


def img_record(name, image, step=None):
    name = tf.constant(name).numpy().decode('utf-8')
    image = np.array(image)
    if image.dtype in (np.float32, np.float64):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    B, T, H, W, C = image.shape
    frames = image.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + '/img', frames, step)
