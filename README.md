# MoDL-ADMM
Python code for MoDL-ADMM reconstruction networks
![banner](https://github.com/8fanmao/MoDL-ADMM/blob/main/Architecture.png)

## Setup Environment
We provide an environment file 
- for Windows: `env_win.yml`
- for Linux: `env_linux.yml`

A new conda environment can be created with 
```
conda env create -f env.yml
```

## Datasets
Several data are provided for demonstration
- Healthy: `data/real`
- BraTS-CEST: `data/simu`

If you want to use your own private data, run `dataset_tfrecord.py`

## Training
An example of network training can be started as follows.
- Stage 1: 
```
python train_net.py --net MoDLap --stage 1 --niter 1 --pre_trained False --num_epoch 10 --batch_size 4
```
- Stage 2:
```
python train_net.py --net MoDLap --stage 2 --niter 5 --pre_trained True --pre_model_id <stage1-ckpt-folder-name> --pre_model_epoch epoch-10 --num_epoch 20 --batch_size 1
```

## Testing
Pre-trained model weights in `models`
An example of network testing can be started as follows.
```
python test_net.py --net MoDLap --stage 2 --niter 10 --pre_model_id <pre-trained-ckpt-folder-name> --pre_mode_epoch epoch-20 --data <tfrecords-folder-name> --acc 4
```

## Acknowledgments
[1] Aggarwal HK, Mani MP, Jacob M. MoDL: Model-based deep learning architecture for inverse problems. IEEE transactions on medical imaging 2018;38(2):394-405.
