import argparse
import os


def str2bool(x):  # turn args.<string> to bool type
    return x.lower() in 'true'


def get_parser_train():
    """
    add console argument for training
    """
    parser = argparse.ArgumentParser(description='Training Configure')

    # model structure setup
    parser.add_argument('--stage', type=int, metavar='int', default=2,
                        help='which stage of train, 1:Nw-only, 2:E2E, '
                             'recommend transfer learning using stage 1 model weihts on stage 2')
    parser.add_argument('--net', type=str, metavar='str', default='MoDLap',
                        help='Net type, '
                             '<MoDL> for original MoDL, '
                             '<MoDLap> for MoDL-ADMM ')
    parser.add_argument('--niter', type=int, metavar='int', default=10,
                        help='number of network iterations')
    parser.add_argument('--denoiser', type=str, metavar='str', default='SK',
                        help='denoiser Nw type, '
                             '<CNN> for plain CNN, '
                             '<SK> for SKnet, ')
    parser.add_argument('--nLayer', type=int, metavar='int', default=5,
                        help='number of CNN/SKnet layers')
    parser.add_argument('--share_weight', type=str, metavar='str', default='Nw',
                        help='<all>: share weights between iteration blocks; '
                             '<Nw>: share Nws weight but lambda and miu varies; '
                             '<none>: all weights varies')
    parser.add_argument('--copy_Nw', type=str2bool, metavar='bool', default=False,
                        help='only useful in <share_weight:none> mode, '
                             'set True to copy Nw_past to Nw_k for faster convergence')
    parser.add_argument('--gradMethod', type=str, metavar='str', default='AG',
                        help='AG for TF auto gradient, MG for manual CG compute, recommend AG when using tf16')
    parser.add_argument('--CG_K', type=int, metavar='int', default=10,
                        help='number of CG steps inside DC block')

    # data setup
    parser.add_argument('--data', type=str, metavar='str', default='data/simu/BraTS-CEST_tfrecords',
                        help='training dataset name')
    parser.add_argument('--frame', type=int, metavar='int', default=2,
                        help='frame num per record')
    parser.add_argument('--acc', type=int, metavar='int', default=4,
                        help='accelerate rate')
    parser.add_argument('--mask_pattern', type=str, metavar='str', default='cartesian',
                        help='mask pattern: e.g. cartesian, radial, spiral')
    parser.add_argument('--N0_fully_sampled', type=str2bool, metavar='bool', default=False,
                        help='whether N0 frame is fully sampled')

    # training setup
    parser.add_argument('--num_epoch', type=int, metavar='int', default=10,
                        help='number of epochs')
    parser.add_argument('--gpu', type=str, metavar='str', default='0',
                        help='GPU No.')
    parser.add_argument('--AMP', type=str2bool, metavar='bool', default=True,
                        help='enable auto mixed precision to save VRAM')
    parser.add_argument('--batch_size', type=int, metavar='int', default=1,
                        help='batch size for each gpu replica')
    parser.add_argument('--CEST_ratio', type=float, metavar='float', default=10.0,
                        help='ratio of L2_CEST in loss func')
    parser.add_argument('--learning_rate', type=float, metavar='float', default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--decay_strategy', type=str, metavar='str', default='cos',
                        help='<exp> for exponential decay, <cos> for cosine decay')
    parser.add_argument('--decay_steps', type=int, metavar='int', default=1700,
                        help='learning rate exp decay step')
    parser.add_argument('--decay_rate', type=float, metavar='float', default=0.9,
                        help='learning rate exp decay rate')
    parser.add_argument('--dev_steps', type=int, metavar='int', default=50,
                        help='step interval for validation')
    parser.add_argument('--save_steps', type=int, metavar='int', default=100,
                        help='step interval for saving ckpt')

    # load ckpt
    parser.add_argument('--pre_trained', type=str2bool, metavar='bool', default=False,
                        help='set True for transfer learning')
    parser.add_argument('--pre_model_id', type=str, metavar='str',
                        default='',
                        help='pre-trained model id')
    parser.add_argument('--pre_model_epoch', type=str, metavar='str', default='',
                        help='pre-trained model epochs, i.e. the ckpt filename')
    parser.add_argument('--pre_log', type=str2bool, metavar='bool', default=False,
                        help='copy pre-trained log or not')
    parser.add_argument('--pre_step', type=int, metavar='int', default=0,
                        help='pre-trained steps for log files')

    return parser


def get_parser_test():
    """
    add console argument for test
    """
    parser = argparse.ArgumentParser(description='Test Configure')

    # model structure setup
    parser.add_argument('--stage', type=int, metavar='int', default=1,
                        help='which stage of train, 1:Nw-only, 2:E2E, '
                             'recommend transfer learning using stage 1 model weihts on stage 2')
    parser.add_argument('--net', type=str, metavar='str', default='MoDLap',
                        help='Net type, '
                             '<MoDL> for original MoDL, '
                             '<MoDLap> for MoDL-ADMM ')
    parser.add_argument('--niter', type=int, metavar='int', default=10,
                        help='number of network iterations')
    parser.add_argument('--denoiser', type=str, metavar='str', default='SK',
                        help='denoiser Nw type, '
                             '<CNN> for plain CNN, '
                             '<SK> for SKnet, ')
    parser.add_argument('--nLayer', type=int, metavar='int', default=5,
                        help='number of CNN/SKnet layers')
    parser.add_argument('--share_weight', type=str, metavar='str', default='Nw',
                        help='<all>: share weights between iteration blocks; '
                             '<Nw>: share Nws weight but lambda and miu varies; '
                             '<none>: all weights varies')
    parser.add_argument('--gradMethod', type=str, metavar='str', default='AG',
                        help='AG for TF auto gradient, MG for manual CG compute, recommend AG when using tf16')
    parser.add_argument('--CG_K', type=int, metavar='int', default=10,
                        help='number of CG steps inside DC block')

    # data setup
    parser.add_argument('--data', type=str, metavar='str', default='data/real/Healthy_tfrecords',
                        help='dataset name')
    parser.add_argument('--real', type=str2bool, metavar='bool', default=True,
                        help='whether test data is real or simulated')
    parser.add_argument('--acc', type=int, metavar='int', default=4,
                        help='accelerate rate')
    parser.add_argument('--mask_pattern', type=str, metavar='str', default='cartesian',
                        help='mask pattern: e.g. cartesian, radial, spiral')
    parser.add_argument('--N0_fully_sampled', type=str2bool, metavar='bool', default=False,
                        help='whether N0 frame is fully sampled')

    # infer setup
    parser.add_argument('--gpu', type=str, metavar='str', default='0',
                        help='GPU No.')
    parser.add_argument('--AMP', type=str2bool, metavar='bool', default=False,
                        help='enable auto mixed precision to save VRAM')
    parser.add_argument('--batch_size', type=int, metavar='int', default=1,
                        help='batch size for each gpu replica')
    parser.add_argument('--CEST_ratio', type=float, metavar='float', default=10.0,
                        help='ratio of L2_CEST in loss func')

    # load ckpt
    parser.add_argument('--pre_model_id', type=str, metavar='str',
                        default='',
                        help='pre-trained model id')
    parser.add_argument('--pre_model_epoch', type=str, metavar='str', default='',
                        help='pre-trained model epochs, i.e. the ckpt filename')
    return parser


def print_options(parser, opt, save_dir):
    """
    print and save options to opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>15}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    file_name = os.path.join(save_dir, 'opt.txt')
    with open(str(file_name), 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


if __name__ == '__main__':
    parser = get_parser_train()
    args = parser.parse_args()
    print(args)
    args.batch_size = 888
    print_options(parser, args, './models/test_args')
    quit()
