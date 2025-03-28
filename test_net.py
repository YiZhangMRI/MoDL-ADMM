import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # ALL INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # NO INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # PRINT ERRO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ONLY FATAL
import csv
import tensorflow as tf
import scipy.io as scio
import numpy as np
import time
from model_net_org import MoDL_Net  # i.e. MoDL-org
from model_net_ADMM import MoDL_Net_ADMM  # i.e. MoDL-ADMM
from model_net_ADMM_TV import MoDL_Net_ADMM_TV  # i.e. MoDL-ADMM with TV sparsity
from dataset_tfrecord import get_dataset
from config_parser import *
from utils.tools import load_mask
from utils.operator import cal_DN_ratio, fft2c_mri
from utils.loss_func import compute_Loss

if __name__ == "__main__":
    # console argument
    parser = get_parser_test()
    args = parser.parse_args()
    mode = 'test'
    multi_coil = True

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    if GPUs:
        print("# list of physical GPUs:", GPUs)
        for GPU in GPUs:
            tf.config.experimental.set_memory_growth(GPU, True)

    # AMP setup
    if args.AMP:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("# AMP enabled with fp16")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print("# AMP disabled with fp32")

    # prepare dataset
    dataset, _ = get_dataset(args.data, args.batch_size, shuffle=False, mode='test')
    data_length = [i for i, _ in enumerate(dataset)][-1] + 1
    tf.print("# test dataset loaded with length %d" % data_length)
    tf.print('dataset:', dataset)

    # prepare under_sampling mask
    mask = load_mask(args.mask_pattern, args.acc)
    if args.N0_fully_sampled:
        mask[0, :, :] = tf.ones([96, 96])
    mask = tf.cast(tf.constant(mask), tf.complex64)
    tf.print("# mask loaded with shape", mask.shape)

    # initialize network
    if args.net == 'MoDL':
        net = MoDL_Net(mask=mask, **vars(args))
    elif args.net == 'MoDLat':
        net = MoDL_Net_ADMM_TV(mask=mask, **vars(args))
    elif args.net == 'MoDLap':
        net = MoDL_Net_ADMM(mask=mask, **vars(args))
    else:
        raise ValueError("Unsupported Network type")
    tf.print("# %s network initialized." % args.net)
    tf.print("using denoiser", args.denoiser, "share weight in", args.share_weight, "mode")

    # load weight
    weight_file = os.path.join('.', 'models', args.pre_model_id, args.pre_model_epoch, 'ckpt')
    net.load_weights(weight_file)
    tf.print("# weight loaded from:", weight_file)

    # result save dir
    if args.real:
        result_dir = os.path.join('results', args.pre_model_id, 'real')
    else:
        result_dir = os.path.join('results', args.pre_model_id, 'simu')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    tf.print("# results will be saved in:", result_dir)

    # set loss func
    compute_Loss = compute_Loss(mask=mask, ratio=args.CEST_ratio, stage=args.stage)
    tf.print("# loss function defined with CEST ratio %.2f" % args.CEST_ratio)

    # record configures
    print_options(parser, args, result_dir)

    Loss_test = []  # cache for test loss
    Loss_mse = []
    Loss_CEST = []
    Loss_TV = []
    Loss_img = []

    # test step
    def test_step(label, csm, mask, stage=2):
        DS_ratio = cal_DN_ratio(label, enable=True)  # calculate data normalization ratio along freq-axis
        k_DS = fft2c_mri(tf.expand_dims(label, axis=1) / DS_ratio * csm)
        k_input = k_DS * mask  # retrospective sub-sampling ksp

        # forward infer
        recon = net(k_input, csm, training=False, stage=stage)
        recon = recon * DS_ratio[:, 0, :, :, :]  # recover strength along freq-axis
        loss_mse, loss_CEST, loss_TV, loss_img, loss_test = compute_Loss(label, recon, batch_size=args.batch_size)
        return recon, label, loss_mse, loss_CEST, loss_TV, loss_img, loss_test

    i = 0
    for i, sample in enumerate(dataset):
    # for sample in dataset.skip(0).take(1):
        if multi_coil:
            label, csm = sample
        else:
            label, _ = sample
            csm = None
        # forward
        tf.print("# processing batch %d ..." % (i + 1))
        t0 = time.time()
        recon, label, loss_mse, loss_CEST, loss_TV, loss_img, loss_test = test_step(label, csm, mask, stage=args.stage)
        if i == 0:
            param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])
            tf.print("# total para num: %d" % param_num)
        t1 = time.time()
        tf.print("forward time %.3f sec" % (t1 - t0))
        tf.print("loss_mse %.3e" % loss_mse)
        tf.print("loss_test %.3e" % loss_test)

        Loss_test = np.append(Loss_test, loss_test.numpy())
        Loss_mse = np.append(Loss_mse, loss_mse.numpy())
        Loss_CEST = np.append(Loss_CEST, loss_CEST.numpy())
        Loss_TV = np.append(Loss_TV, loss_TV.numpy())
        Loss_img = np.append(Loss_img, loss_img.numpy())

        # save mat
        result_file = os.path.join(result_dir, 'recon_' + str(i + 1) + '.mat')
        datadict = {'recon': np.squeeze(tf.transpose(recon, [0, 2, 3, 1]).numpy()),
                    'label': np.squeeze(tf.transpose(label, [0, 2, 3, 1]).numpy()),
                    'csm': np.squeeze(tf.transpose(csm[0, :, 0, :, :], [1, 2, 0]).numpy())}
        scio.savemat(result_file, datadict)

        # record loss to csv
        item = {'Batch': i + 1,
                'Loss_test': loss_test.numpy(),
                'Loss_CEST': loss_CEST.numpy(),
                'Loss_TV': loss_TV.numpy(),
                'Loss_img': loss_img.numpy(),
                'Loss_mse': loss_mse.numpy()}
        fieldnames = ['Batch', 'Loss_test', 'Loss_CEST', 'Loss_TV', 'Loss_img', 'Loss_mse']
        with open(os.path.join(result_dir, 'test_loss.csv'), mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not os.path.getsize(os.path.join(result_dir, 'test_loss.csv')):
                writer.writeheader()
            writer.writerows([item])

    with open(os.path.join(result_dir, 'test_loss.txt'), 'w') as file:
        for loss_item in Loss_test:
            file.write(f'{loss_item}\n')
