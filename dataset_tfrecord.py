import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # ALL INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # NO INFO
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # PRINT ERRO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ONLY FATAL
import tensorflow as tf
import re
import random
import scipy.io as scio
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt
from utils.espirit import espirit, fft
from utils.operator import cal_DN_ratio, ifft2c_mri, fft2c_mri
from utils.tools import seed_everything, mycopyfile


def gen_csm_espirit(img_coil, ismask=False):
    """ generating sensitivity map with ESPIRiT """
    # nb nc nz nx ny
    img_coil = tf.transpose(img_coil, perm=[0, 2, 3, 4, 1])
    [nb, nz, nx, ny, nc] = img_coil.shape
    csm = np.zeros([nb, 1, nx, ny, nc]).astype(np.complex64)

    for p in range(img_coil.shape[0]):
        tf.print('processing patient ' + str(p + 1))
        x = np.squeeze(img_coil[p, 0, :, :, :])  # use the 1st frame inf ppm
        x = np.expand_dims(x, axis=0)

        # Derive ESPIRiT operator
        if ismask:
            A = np.angle(x)
            x = np.power(np.abs(x), 1.5)
            x = (np.cos(A) + 1j * np.sin(A)) * x
            X = fft(x, (0, 1, 2))  # input X to be kspace
            esp = espirit(X, 12, 48, 0.02, 0.9)  # mask mode：0.9, create CSM partial
        else:
            X = fft(x, (0, 1, 2))
            esp = espirit(X, 12, 48, 0.02, 0.0)  # fullmode
        # esp [nz nx ny nc nc]
        for idx in range(nc):
            csm[p, :, :, :, idx] = np.squeeze(esp[:, :, :, idx, 0])

    csm = tf.transpose(csm, perm=[0, 4, 1, 2, 3])
    csm = np.repeat(csm, nz, axis=2)
    return csm


def compress(APTimage):
    """ compress 54 frames CEST image into 32 by averaging """
    repeat = [1, 1, 1, 1, 2, 2, 6, 2, 2, 2, 2, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 6, 2, 2, 1, 1, 1]
    r = 0
    f = 0
    APTimage_c = np.zeros([96, 96, 32], dtype=APTimage.dtype)
    while f < 54:
        if repeat[r] == 1:
            APTimage_c[:, :, r] = APTimage[:, :, f]
            f += 1
        else:
            for rf in range(repeat[r]):
                APTimage_c[:, :, r] += APTimage[:, :, f]
                f += 1
            APTimage_c[:, :, r] /= repeat[r]
        r += 1
    return APTimage_c


def amp_normalize_minmax(data):
    """ normalization with amplitude max-min """
    # nb nt nx ny
    assert data.dtype == 'complex64', "data is not complex64"
    real = data.real
    imag = data.imag
    Ampmax = np.max(np.abs(data))
    Ampmin = np.min(np.abs(data))
    # assuming Amp_min → 0
    real = real / (Ampmax - Ampmin)
    imag = imag / (Ampmax - Ampmin)
    data = real + 1j * imag
    return data


def ksp_normalize_minmax(data):
    """ normalization with ksp center ACS """
    # nb nt nx ny
    assert data.dtype == 'complex64', "data is not complex64"
    [_, _, nx, ny] = np.shape(data)
    mask = np.zeros([nx, ny])
    # auto calibration line 12/6
    mask[nx // 2 - 6:nx // 2 + 6, :] = 1
    # mask[nx // 2 - 3:nx // 2 + 3, :] = 1
    data_smooth = ifft2c_mri(fft2c_mri(data) * mask)
    real = data.real
    imag = data.imag
    Ampmax = np.max(np.abs(data_smooth))
    Ampmin = np.min(np.abs(data_smooth))
    # assuming Amp_min → 0
    real = real / (Ampmax - Ampmin)
    imag = imag / (Ampmax - Ampmin)
    data = real + 1j * imag
    return data


def _int64_feature(value):
    """ data format int64 """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


def _bytes_feature(value):
    """ data format bytes """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float32_feature(value):
    """ data format float32 """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def serialize_data(image_raw, csm_raw):
    """ data serialization to bytes """
    feature = {
        'label_real': _float32_feature(np.array(tf.math.real(image_raw))),
        'label_imag': _float32_feature(np.array(tf.math.imag(image_raw))),
        'csm_real': _float32_feature(np.array(tf.math.real(csm_raw))),
        'csm_imag': _float32_feature(np.array(tf.math.imag(csm_raw))),
        'img_shape': _int64_feature(np.array(image_raw.shape)),
        'csm_shape': _int64_feature(np.array(csm_raw.shape))
    }
    patient_data = tf.train.Example(features=tf.train.Features(feature=feature))

    return patient_data.SerializeToString()


def parse_function(example_proto):
    """ parser for .tfrecord files """
    dics = {
        'label_real': tf.io.VarLenFeature(dtype=tf.float32),
        'label_imag': tf.io.VarLenFeature(dtype=tf.float32),
        'csm_real': tf.io.VarLenFeature(dtype=tf.float32),
        'csm_imag': tf.io.VarLenFeature(dtype=tf.float32),
        'img_shape': tf.io.FixedLenFeature([3, ], dtype=tf.int64),
        'csm_shape': tf.io.FixedLenFeature([4, ], dtype=tf.int64)
    }

    parsed_example = tf.io.parse_single_example(example_proto, dics)
    parsed_example['label_real'] = tf.sparse.to_dense(parsed_example['label_real'])
    parsed_example['label_imag'] = tf.sparse.to_dense(parsed_example['label_imag'])
    parsed_example['csm_real'] = tf.sparse.to_dense(parsed_example['csm_real'])
    parsed_example['csm_imag'] = tf.sparse.to_dense(parsed_example['csm_imag'])

    label = tf.complex(parsed_example['label_real'], parsed_example['label_imag'])
    csm = tf.complex(parsed_example['csm_real'], parsed_example['csm_imag'])

    label = tf.reshape(label, parsed_example['img_shape'])
    csm = tf.reshape(csm, parsed_example['csm_shape'])

    return label, csm


def mat_to_tfrecords(MatLocation, DataType, DataCap_l, DataCap_h, TargetName, Prefix='', framePerRecord=16,
                     compress=False):
    """
    encode .mat to .tfrecords
    MatLocation: root of the dataset
    DataType: subdirectory of the dataset
    DataCap_l & DataCap_l: patient data ids
    TargetName: dir for saving tfrecords
    """
    tf.print('# converting .mat in', MatLocation, 'to .tfrecords saved in', TargetName)
    record = 0
    if not os.path.isdir(TargetName):
        os.makedirs(TargetName)

    for DT in range(len(DataType)):
        # scan and record file name of .mat into lists
        dataLocation = os.path.join(MatLocation, DataType[DT])
        tf.print('# moving to', dataLocation)
        patientIdList = []  # patient id name list
        aptMatList = []  # apt data name list
        csmMatList = []  # csm data name list
        frameNumList = [0]  # frame num list
        for DC in range(DataCap_l[DT], DataCap_h[DT] + 1):
            # patientLocation = os.path.join(dataLocation, str(DC))  # fastMRI-CEST id
            patientLocation = os.path.join(dataLocation, str(DC).zfill(3))  # BraTS-CEST id %3d
            if not os.path.isdir(patientLocation):
                tf.print('file %s is empty' % str(DC).zfill(3))
                continue
            else:
                frameNum = 0
                for MatName in os.listdir(patientLocation):
                    if re.match('CEST' + r'.*' + '.mat' + '$', MatName):
                        aptMatList.append(os.path.join(patientLocation, MatName))
                        frameNum += 1
                    if re.match('espirit' + r'.*' + '.mat' + '$', MatName):
                        csmMatList.append(os.path.join(patientLocation, MatName))
                tf.print('scan file %s with %d frames' % (str(DC).zfill(3), frameNum))
                patientIdList.append(str(DC).zfill(3))
                frameNumList.append(frameNumList[-1] + frameNum)

        """ if you want to have exactly frame for every tfrecord """
        recordNum = floor(len(aptMatList) / framePerRecord)  # num of .tfrecords for current DT
        recordFrame = [framePerRecord] * recordNum
        """ elif you want to make full use of data """
        # recordNum = ceil(len(aptMatList) / framePerRecord)  # num of .tfrecords for current DT
        # recordFrame = [framePerRecord] * (recordNum - 1) + [len(aptMatList) - framePerRecord * (recordNum - 1)]

        # write tfrecords
        frameNum = 0
        patient = 0
        for RN in range(recordNum):
            record += 1
            tfrName = os.path.join(TargetName, Prefix + str(record).zfill(4) + '.tfrecords')
            with tf.io.TFRecordWriter(tfrName) as writer:
                tf.print('write into record %s' % str(record).zfill(4))
                for AC in range(recordFrame[RN]):
                    frameNum += 1
                    if frameNum > frameNumList[patient]:
                        tf.print('# processing %s patient %s' % (DataType[DT], str(patientIdList[patient]).zfill(3)))
                        patient += 1
                    aptMatName = aptMatList[framePerRecord * RN + AC]
                    csmMatName = csmMatList[framePerRecord * RN + AC]
                    image_raw, csm_raw = patient_data(aptMatName, csmMatName, compress=compress)
                    serialized_patient_data = serialize_data(image_raw, csm_raw)
                    writer.write(serialized_patient_data)
    tf.print('# .tfrecords transformation finished')


def patient_data(aptMatName, csmMatName, compress=False):
    """
    load CEST img & CSM data of each patient
    compress: set True to compress 54 offsets into 32 through averaging
    """
    APTimage = np.complex64(scio.loadmat(aptMatName)['APT_data'])  # 96x96x54 APT image
    # APTimage = np.flip(APTimage, axis=0)  # flipud for fastmri data
    if compress:
        APTimage = compress(APTimage)  # 96x96x32 APT image
    APTimage = np.expand_dims(APTimage.transpose(2, 0, 1), axis=0)  # 1x54x96x96 APT image
    """ normalization using fully sampled data amplitude """
    # APTimage = amp_normalize_minmax(APTimage)
    """ normalization using ksp center data amplitude i.e. ACS """
    APTimage = ksp_normalize_minmax(APTimage)
    CSM = np.complex64(scio.loadmat(csmMatName)['sensitivities'])  # 96x96x16 CSM
    CSM = np.expand_dims(CSM.transpose(2, 0, 1), axis=1)  # 16x1x96x96 CSM
    APTimage = np.squeeze(APTimage)  # 54x96x96 APT image
    return APTimage, CSM


def gen_dataset_loc(dataset_name, mode='train', stage='load', ratio=[0.98, 0.01, 0.01]):
    """
    generate dataset filenames
    mode: train / dev / test
    stage: 'load' for loading data and 'gen' for randomly dividing data into train / dev / test according to ratio
    """
    assert os.path.isdir(dataset_name), "data location doesnt exist"
    if stage == 'load':
        assert os.path.isfile(os.path.join(dataset_name, 'train.txt')) \
               or os.path.isfile(os.path.join(dataset_name, 'test.txt')), "data dic doesnt exist"

        def read_txt(txtname):
            filenames = []
            f = open(os.path.join(dataset_name, txtname), "r", encoding='utf-8')
            line = f.readline()
            while line:
                filenames.append(line.strip().split('\n')[0])
                line = f.readline()
            f.close()
            return filenames

        if mode == 'train':
            # load train dataset name
            filenames = read_txt('train.txt')
            # randomly shuffle different mode of image (FLAIR/T1/T2)
            random.shuffle(filenames)
        elif mode == 'dev':
            # load dev dataset name
            filenames = read_txt('dev.txt')
        elif mode == 'test':
            # load test dataset name
            filenames = read_txt('test.txt')
        return filenames

    elif stage == 'gen':
        # gen dataset name randomly
        def txt_filter(f):
            if f[-4:] in ['.txt']:
                return False
            else:
                return True

        files = os.listdir(dataset_name)
        files = list(filter(txt_filter, files))  # do not include txt file
        patientnum = len(files)
        [dev_num, test_num] = [int(i * patientnum) + 1 for i in ratio[1:3]]
        num = random.sample(range(1, patientnum + 1), dev_num + test_num)
        dev_filenames = [files[i - 1] for i in num[0:dev_num]]
        test_filenames = [files[i - 1] for i in num[-test_num:]]

        def num_filter(f, num):
            if int(f[-14:-10]) in num:  # find the ids of file
                return False
            else:
                return True

        train_filenames = list(filter(lambda x: num_filter(x, num), files))
        train_filenames.sort(key=lambda x: int(x[-14:-10]))
        dev_filenames.sort(key=lambda x: int(x[-14:-10]))
        test_filenames.sort(key=lambda x: int(x[-14:-10]))
        with open(os.path.join(dataset_name, 'train.txt'), 'w') as file:
            for filename in train_filenames:
                file.write(f'{filename}\n')
        with open(os.path.join(dataset_name, 'dev.txt'), 'w') as file:
            for filename in dev_filenames:
                file.write(f'{filename}\n')
        with open(os.path.join(dataset_name, 'test.txt'), 'w') as file:
            for filename in test_filenames:
                file.write(f'{filename}\n')


def get_dataset(dataset_name, batch_size, shuffle=False, mode='train', framePerRecord=16):
    """ load & decode .tfrecord files into mini-batches """
    filenames = [os.path.join(dataset_name, f) for f in gen_dataset_loc(dataset_name, mode)]
    # load tfrecords with file name list
    dataset = tf.data.TFRecordDataset(filenames)
    tf.print('all files in', dataset_name, 'is loaded')
    # tf.print('raw record:', dataset)

    # parse tfrecords
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    tf.print('finish decoding...')

    # shuffle and get mini-batches
    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=20)
        tf.print('shuffling...')
    parsed_dataset = parsed_dataset.batch(batch_size)
    tf.print('batch created')

    # cal data length
    data_len = floor(len(filenames) * framePerRecord / batch_size)
    return parsed_dataset.prefetch(batch_size), data_len


if __name__ == "__main__":
    """ GPU setup """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    """ location of .mat data """
    MatLocation = os.path.join('.', 'data', 'simu')
    TargetName = os.path.join('.', 'data', 'simu', 'BraTS-CEST_tfrecords')
    Prefix = 'brats_'
    DataType = ['BraTS-CEST']
    # MatLocation = os.path.join('.', 'data', 'real')
    # TargetName = os.path.join('.', 'data', 'real', 'Healthy_tfrecords')
    # Prefix = 'healthy_'
    # DataType = ['Healthy']
    DataCap_l = [1]
    DataCap_h = [1]

    """ 
    Encode tfrecord files
    compress [True]: compress repeated offsets 54 frames to 32 frames
    noise [True]: Add gaussian noise at std <level> by scaled MT ratio <alpha*(1-Z)^alpha>
    """
    mat_to_tfrecords(MatLocation, DataType, DataCap_l, DataCap_h, TargetName, Prefix, compress=False, framePerRecord=1)

    """ randomly generate train/val/test txt"""
    gen_dataset_loc(TargetName, stage='gen', ratio=[0.5, 0.25, 0.25])

    """ decode tfrecord files and gen dataset """
    # seed_everything(seed=11)
    Recon, length = get_dataset(TargetName, batch_size=1, shuffle=False, mode='test', framePerRecord=1)
    tf.print("load data in %s with length %d" % (TargetName, length))

    """ show some demo """
    batch = 0
    # for step, rec in enumerate(Recon):
    for rec in Recon.skip(0).take(1):
        batch += 1
        img, csm = rec
        DS_ratio = cal_DN_ratio(img, enable=True)
        ksp = fft2c_mri(img)
        tf.print('batch', batch)
        tf.print('img', img.shape)
        tf.print('csm', csm.shape)
        tf.print('ksp', ksp.shape)
        tf.print('img min', np.min(np.abs(img)))
        tf.print('img max', np.max(np.abs(img)))
        tf.print('csm min', np.min(np.abs(csm)))
        tf.print('csm max', np.max(np.abs(csm)))

        for f in range(32):
            plt.subplot(4, 8, f + 1)
            plt.imshow(np.squeeze(tf.abs(img[:, f, :, :] / DS_ratio[:, :, f, :, :])), cmap='jet', vmin=0.0, vmax=1.0)
            plt.axis('off')
        plt.show()
