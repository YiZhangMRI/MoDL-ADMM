import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # ALL INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # NO INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # PRINT ERRO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ONLY FATAL
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from glob import glob
from datetime import datetime
from tqdm import tqdm
from model_net_org import MoDL_Net  # i.e. MoDL-org
from model_net_ADMM import MoDL_Net_ADMM  # i.e. MoDL-ADMM
from model_net_ADMM_TV import MoDL_Net_ADMM_TV  # i.e. MoDL-ADMM with TV sparsity
from dataset_tfrecord import get_dataset
from utils.tools import img_record, load_mask, mycopyfile, seed_everything, Warmingup_ExpDecay, Warmingup_CosDecay
from utils.operator import cal_DN_ratio, fft2c_mri, noisy_sample
from utils.loss_func import compute_Loss
from config_parser import *

if __name__ == "__main__":
    # console argument
    parser = get_parser_train()
    args = parser.parse_args()
    mode = 'training'
    multi_coil = True
    # initial random seed
    seed_everything(seed=11)

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    if GPUs:
        print("# list of physical GPUs:", GPUs)
        for GPU in GPUs:
            tf.config.experimental.set_memory_growth(GPU, True)
            # # Create 4 virtual GPUs with 6GB memory each to test multi-GPU parallel compute
            # try:
            #     tf.config.set_logical_device_configuration(
            #         GPUs[0],
            #         [tf.config.LogicalDeviceConfiguration(memory_limit=6144),
            #          tf.config.LogicalDeviceConfiguration(memory_limit=6144),
            #          tf.config.LogicalDeviceConfiguration(memory_limit=6144),
            #          tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
            #     logical_GPUs = tf.config.list_logical_devices('GPU')
            #     print(len(GPUs), "Physical GPU,", len(logical_GPUs), "Logical GPUs")
            #     print(logical_GPUs)
            # except RuntimeError as e:
            #     # Virtual devices must be set before GPUs have been initialized
            #     print(e)

    # AMP setup
    if args.AMP:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("# AMP enabled with fp16")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print("# AMP disabled with fp32")

    # distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    print("# Number of devices: {}".format(num_replicas))
    batch_size_global = num_replicas * args.batch_size

    # prepare dataset
    dataset, data_length = get_dataset(args.data, batch_size_global, shuffle=True, mode='train',
                                       framePerRecord=args.frame)
    tf.print("# training dataset loaded with length %d" % data_length)
    tf.print('dataset:', dataset)
    dev_bsize = 1
    devdata, devdata_length = get_dataset(args.data, dev_bsize, shuffle=False, mode='dev',
                                          framePerRecord=args.frame)
    # devdata_length = [i for i, _ in enumerate(devdata)][-1] + 1
    tf.print("# dev dataset loaded with length %d" % devdata_length)
    tf.print('dev dataset:', devdata)
    # distribute data to replica
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # log save configures
    logdir = os.path.join('.', 'logs')
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    model_id = TIMESTAMP + '_' + args.net + '_stage_%s' % args.stage + '_niter_%s' % args.niter + '_sw_%s' % args.share_weight
    model_id += '_' + args.denoiser + '_%s' % args.nLayer
    if args.decay_strategy == 'exp':
        lr_id = '_lr_%s' % args.learning_rate + '_decay_%s' % args.decay_rate + '_per_%s' % args.decay_steps
    elif args.decay_strategy == 'cos':
        lr_id = '_lr_%s' % args.learning_rate + '_decay_cos' + '_stp_%s' % data_length
    else:
        raise ValueError("Unsupported lr decay strategy")
    model_id += '_acc_%s' % args.acc + '_' + args.mask_pattern + '_batch_%s' % batch_size_global + lr_id
    with strategy.scope():
        summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, model_id))

    # model weight save configures
    modeldir = os.path.join('.', 'models', model_id)
    os.makedirs(modeldir)
    # sav config to txt
    print_options(parser, args, modeldir)

    # prepare under_sampling mask
    mask = load_mask(args.mask_pattern, args.acc)
    if args.N0_fully_sampled:
        mask[0, :, :] = tf.ones([96, 96])
    mask = tf.cast(tf.constant(mask), tf.complex64)
    tf.print("# mask loaded with shape", mask.shape)

    # if args.stage > 1:
    #     assert args.pre_trained, "ERROR! stage2 must be initialized with pre-trained stage1 weights"
    # if args.stage < 2 or args.share_weight.lower() != 'none':
    #     args.copy_Nw = False

    # initialize network
    with strategy.scope():
        if args.net == 'MoDL':
            net = MoDL_Net(mask=mask, **vars(args))
        elif args.net == 'MoDLat':
            net = MoDL_Net_ADMM_TV(mask=mask, **vars(args))
        elif args.net == 'MoDLap':
            net = MoDL_Net_ADMM(mask=mask, **vars(args))
        else:
            raise ValueError("Unsupported Network type")
    tf.print("# %s network initialized." % args.net)
    tf.print('using denoiser', args.denoiser, 'share weight in', args.share_weight, 'mode')

    # load pre-trained weight
    if args.pre_trained:
        weight_file = os.path.join('.', 'models', args.pre_model_id, args.pre_model_epoch, 'ckpt')
        net.load_weights(weight_file)
        tf.print("# pre-trained weight loaded from", args.pre_model_id, args.pre_model_epoch)
        if args.pre_log:  # copy pre-trained logs
            preLog = os.path.join(logdir, 'training', args.pre_model_id, '*')
            dst_dir = os.path.join(logdir, mode, model_id)
            src_file_list = glob(preLog)
            for srcfile in src_file_list:
                mycopyfile(srcfile, dst_dir)
            tf.print("# pre-trained log copied")
    else:
        tf.print("# model will be randomly initialized")
        args.pre_step = 0

    # set learning rate & optimizer
    with strategy.scope():
        if args.decay_strategy == 'exp':
            learning_rate_schedule = Warmingup_ExpDecay(
                learning_rate=args.learning_rate, warm_steps=100, decay_steps=float(args.decay_steps),
                decay_rate=args.decay_rate)
            tf.print('max lr', args.learning_rate, 'decay with', args.decay_strategy, 'rate', args.decay_rate, 'every',
                     args.decay_steps, 'steps')
        elif args.decay_strategy == 'cos':
            learning_rate_schedule = Warmingup_CosDecay(
                learning_rate=args.learning_rate, warm_steps=100, epochs=args.num_epoch, steps=float(data_length))
            tf.print('max lr', args.learning_rate, 'decay with', args.decay_strategy, data_length,
                     'steps per epoch')
        optimizer = tf.optimizers.Adam(learning_rate_schedule, clipvalue=0.1)
        # AMP scaled optimizer
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        tf.print("# optimizer and lr initialized")

    # set loss func
    with strategy.scope():
        compute_Loss = compute_Loss(mask=mask, ratio=args.CEST_ratio, stage=args.stage)
    tf.print("# loss function defined with CEST ratio %.2f" % args.CEST_ratio)

    # set training step for each replica
    def train_step(label, csm, mask, stage=2):
        DN_ratio = cal_DN_ratio(label, enable=True)  # calculate data normalization ratio along freq-axis
        k_DS = fft2c_mri(tf.expand_dims(label, axis=1) / DN_ratio * csm)
        k_input = noisy_sample(k_DS, mask, sigma=0.0, ratio=DN_ratio)  # retrospective sub-sampling ksp
        # tf.print('label', label.shape, 'csm', csm.shape, 'mask', mask.shape, 'k_input', k_input.shape)
        # scio.savemat('./input.mat', {'label': label.numpy(), 'csm': csm.numpy(), 'k_input': k_input.numpy()})

        with tf.GradientTape() as tape:
            # forward prop
            recon = net(k_input, csm, training=True, stage=stage)
            recon = recon * DN_ratio[:, 0, :, :, :]  # recover strength along freq-axis
            # <L_mse, L_CEST, L_TV, L_L2> are metric for logs, optimization relies on <L_train>
            loss_mse, loss_CEST, loss_TV, loss_L2, loss_train = compute_Loss(label, recon, batch_size=batch_size_global)
            scaled_loss_train = optimizer.get_scaled_loss(loss_train)  # AMP scaling
        # backward prop
        scaled_grads = tape.gradient(scaled_loss_train, net.trainable_weights)  # AMP scaling
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        grads_cl, global_norm = tf.clip_by_global_norm(grads, clip_norm=1000.0)  # gradient clipping
        global_norm = tf.cond(
            tf.math.logical_or(tf.experimental.numpy.isinf(global_norm), tf.experimental.numpy.isnan(global_norm)),
            lambda: 65535.0 * batch_size_global, lambda: global_norm)  # scale grad norm for log
        optimizer.apply_gradients(zip(grads_cl, net.trainable_weights))
        return loss_mse, loss_CEST, loss_TV, loss_L2, loss_train, global_norm


    @tf.function
    def distributed_train_step(label, csm, mask, stage=2):
        loss_mse, loss_CEST, loss_TV, loss_L2, loss_train, global_norm = strategy.run(
            train_step, args=(label, csm, mask, stage)
        )
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss_mse, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.SUM, loss_CEST, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.SUM, loss_TV, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.SUM, loss_L2, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.SUM, loss_train, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.SUM, global_norm, axis=None)


    # validation step
    @tf.function
    def val_step(dlabel, dcsm, mask, stage=2):
        DS_ratio = cal_DN_ratio(dlabel, enable=True)
        dk_DS = fft2c_mri(tf.expand_dims(dlabel, axis=1) / DS_ratio * dcsm)
        dk_input = noisy_sample(dk_DS, mask, sigma=0.0)
        # forward
        drecon = net(dk_input, dcsm, training=False, stage=stage)
        drecon = drecon * DS_ratio[:, 0, :, :, :]
        loss_mse, _, _, _, loss_dev = compute_Loss(dlabel, drecon, batch_size=dev_bsize)
        return dlabel, drecon, loss_mse, loss_dev


    # Iterate over epochs
    total_step = 0
    loss_list = []  # cache for training loss
    dev_loss_list = []
    corrupted_data = 0  # record corrupted data

    for epoch in range(args.num_epoch):
        if epoch != 0:
            # re-shuffle dataset every epoch
            dataset.shuffle(buffer_size=50)
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            tf.print('# training dataset re-shuffled')
        try:  # avoid <data_loss_error> when yielding sample on hardware w/o ECC RAM
            with tqdm(enumerate(dist_dataset), ascii=True, total=data_length) as tepoch:
                tepoch.set_description("# Epoch %d / %d" % (epoch + 1, args.num_epoch))
                for step, sample in tepoch:
                    try:  # avoid <data_loss_error> when sample is corrupted
                        if step == 0:
                            tqdm.write("======= new epoch =======")
                        if multi_coil:
                            label, csm = sample
                        else:
                            label, _ = sample
                            csm = None
                        # optimization on mini-batch
                        loss_mse, loss_CEST, loss_TV, loss_L2, loss_train, global_norm = distributed_train_step(
                            label, csm, mask, stage=args.stage)
                        global_norm = global_norm / batch_size_global
                        lr = learning_rate_schedule(total_step)
                        loss_list = np.append(loss_list, loss_train)

                        # calculate parameter number
                        if total_step == 0:
                            param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])
                            tqdm.write('# net total trainable param num %d' % param_num)

                        # log output
                        tepoch.set_postfix({"lr": "%.3e" % lr,
                                            "mse": "%.3e" % loss_mse,
                                            "loss": "%.3e" % loss_train,
                                            "gard": "%.3e" % global_norm})

                        # write log into file
                        with summary_writer.as_default():
                            tf.summary.scalar('loss/img_mse', loss_mse, step=total_step + args.pre_step)
                            tf.summary.scalar('loss/img_L2', loss_L2, step=total_step + args.pre_step)
                            tf.summary.scalar('loss/CEST_L2', loss_CEST, step=total_step + args.pre_step)
                            tf.summary.scalar('loss/TV_L1', loss_TV, step=total_step + args.pre_step)
                            tf.summary.scalar('loss/Loss_train', loss_train, step=total_step + args.pre_step)
                            tf.summary.scalar('grad/global_norm', global_norm, step=total_step + args.pre_step)
                            tf.summary.scalar('lr', lr, step=total_step + args.pre_step)

                        # try validation data every dev_steps
                        if step % args.dev_steps == 0 and step > 0:
                            loss_mse_dev = 0.0
                            loss_dev = 0.0
                            for dev_sample in devdata.skip(random.randint(0, devdata_length - 2)).take(2):
                                # randomly pick 2 validation data
                                if multi_coil:
                                    dev_label, dev_csm = dev_sample
                                else:
                                    dev_label, _ = dev_sample
                                    dev_csm = None
                                dev_label, dev_recon, loss_md, loss_d = val_step(dev_label, dev_csm, mask,
                                                                                 stage=args.stage)
                                loss_mse_dev += loss_md
                                loss_dev += loss_d
                            loss_mse_dev /= 2
                            loss_dev /= 2
                            tqdm.write("Dev loss = %.3e mse = %.3e at Epoch %d Step %d" % (
                                loss_dev, loss_mse_dev, epoch + 1, step))
                            dev_loss_list = np.append(dev_loss_list, loss_dev)
                            # record dev_loss
                            with summary_writer.as_default():
                                tf.summary.scalar('loss/Loss_dev', loss_dev, step=total_step + args.pre_step)
                                tf.summary.scalar('loss/Loss_mse_dev', loss_mse_dev, step=total_step + args.pre_step)

                        # save model and img every save_steps
                        if step % args.save_steps == 0 and step > 0:
                            model_epoch_dir = os.path.join(modeldir, 'epoch-' + str(epoch + 1) + '_step-' + str(step),
                                                           'ckpt')
                            net.save_weights(model_epoch_dir, save_format='tf')
                            tqdm.write("# model weight saved at Epoch %d Step %d" % (epoch + 1, step))

                            # record img to tensorboard
                            with summary_writer.as_default():
                                combine_img = tf.concat(
                                    [tf.abs(dev_label)[0:1, :, :, :], tf.abs(dev_recon)[0:1, :, :, :]],
                                    axis=0).numpy()
                                combine_img = np.expand_dims(combine_img, -1)
                                img_record('result', combine_img, step=total_step + args.pre_step)

                        total_step += 1

                    except tf.errors.DataLossError as erro:
                        tqdm.write("### corrupted record ###")
                        tqdm.write(erro)
                        tqdm.write("### skip batch %d ###" % step)
                        corrupted_data += 1
                    if corrupted_data > 0:
                        tqdm.write("### now corrupted: %d data at epoch %d" % (corrupted_data, epoch))
        except tf.errors.DataLossError as erro:
            tf.print("### corrupted record ###")
            tf.print(erro)
            tf.print("### skip epoch %d ###" % epoch)
            corrupted_data += 1

        # save model each epoch
        model_epoch_dir = os.path.join(modeldir, 'epoch-' + str(epoch + 1), 'ckpt')
        net.save_weights(model_epoch_dir, save_format='tf')
        tf.print("# model weight saved at Epoch %d" % (epoch + 1))

    # save loss to txt and plot img
    print("# saving loss to txt")
    with open(os.path.join(modeldir, 'training_loss.txt'), 'w') as file:
        for loss_item in loss_list:
            file.write(f'{loss_item}\n')
    with open(os.path.join(modeldir, 'dev_loss.txt'), 'w') as file:
        for loss_item in dev_loss_list:
            file.write(f'{loss_item}\n')

    # plt.title('Model Loss')
    # xaxis = np.arange(0, loss_list.shape[0], 1)
    # sub_xaxis = np.arange(0, loss_list.shape[0], loss_list.shape[0] / dev_loss_list.shape[0])
    # plt.semilogy(xaxis, loss_list, label="Train_Loss")
    # plt.semilogy(sub_xaxis, dev_loss_list, label="Dev_Loss")
    # label = ["Train_Loss", "Dev_Loss"]
    # plt.legend(label, loc=1)
    # plt.savefig(os.path.join(modeldir, 'Loss.png'))
    # plt.show()

    print("# Loss saved and training finished :)")
