Implementation of ICASSP2021 paper: VISUAL PRIVACY PROTECTION VIA MAPPING DISTORTION. This repository mainly contains the code for image classification with CIFAR-10 dataset.

## Preparation

- python3.6.2, pytorch1.7.0
```bash
pip install -r requirements.txt
```

## Generating the Modified CIFAR-10 Dataset

- Download [pretrained model](https://drive.google.com/file/d/1BVPlp5ory1smQOD1GKB5Gop2cqGHT7m2/view?usp=sharing) from Google Drive and put it in the folder "/path/to/project/runs/train_original_cifar10/checkpoints/".

- Download original [cifar10 training images](https://drive.google.com/file/d/1HXPsYtSQ-7cXYtk96rOMw4P9znKpsvzP/view?usp=sharing) and original [cifar10 testset](https://drive.google.com/file/d/1Ej1kKPv0KWte32l-qUIUD51naSD782Pv/view?usp=sharing) from Google Drive, put them in the folder "/path/to/project/data" and unzip them.
```bash
unzip cifar10_ori_images.zip
unzip testdir.zip
```



- Run the script below to start the generation process for MDP.

```bash
bash generate_modified_cifar10.sh /path/to/project/experiments/generate_modified_cifar10_resnet.yaml
```

- Check modified cifar10 locates at '/path/to/project/data/generate_modified_cifar10_resnet/'.

- Run the script below to start the generation process for AugMDP (T=2).

```bash
bash generate_modified_cifar10.sh /path/to/project/experiments/generate_modified_cifar10_resnet_multi2.yaml
```

- Check modified cifar10 (T=2) locates at '/path/to/project/data/generate_modified_cifar10_resnet_multi2/'.


## Training on the Modified CIFAR-10 Dataset

- Run the script below to start the training process for MDP.

```bash
bash train_modified_cifar10.sh /path/to/project/experiments/train_modified_cifar10.yaml
```

- Run the script below to start the training process for AugMDP (T=2).

```bash
bash train_modified_cifar10.sh /path/to/project/experiments/train_modified_cifar10_multi2.yaml
```


## Acknowledgement

Thanks to MadryLab (https://github.com/MadryLab/constructed-datasets) for their inspiration. 
