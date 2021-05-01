'''
Code for generating modified dataset for CIFAR-10.
Copyright (c) Peidong Liu, 2020
'''
import argparse
import os
import yaml
import time
import random
import glob
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as ours_models_cifar
from torchvision.transforms import ToPILImage
from utils import PGD


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config', default='', type=str)
args = parser.parse_args()

config_path = args.config
assert config_path != '', 'you haven`t assign any config file yet!'
config = yaml.load(open(config_path))
if config['MODEL']['TORCHVISION']:
    models = torchvision.models
else:
    models = ours_models_cifar
# Validate dataset
dataset = config['DATA']['DATASET']

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
use_cuda = torch.cuda.is_available()

# Random seed
seed = config['SEED']
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)
rootdir = config['ROOT_DIR']

def main():

    # Data
    print('==> Preparing dataset %s' % dataset, flush=True)
    traindir = os.path.join(rootdir, 'data/cifar10_ori_images', 'train')

    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])),
        batch_size=config['TRAIN']['BATCH_SIZE'], shuffle=True,
        num_workers=config['DATA']['NUM_WORKERS'], pin_memory=True)

    num_classes = 10
    # Model
    model_type = config['MODEL']['TYPE']
    print("==> creating model '{}'".format(model_type))
    if config['MODEL']['TORCHVISION']:
        model = models.__dict__[model_type](
                    num_classes=num_classes,
                    pretrained=config['TRAIN']['PRETRAINED'],
                )
    else:
        if model_type.startswith('resnet50'):
            model = models.ResNet50()
        elif model_type.startswith('resnet152'):
            model = models.ResNet152()
        elif model_type.startswith('resnet_gen'):
            model = models.__dict__[model_type](
                num_classes=num_classes,
                depth=config['MODEL']['DEPTH'],
                )
        elif model_type.startswith('densenet154'):
            model = models.densenet(depth=154)
        else:
            assert False, 'not supported model!'

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0), flush=True)

    # resume from trained model
    if (config['CHECKPOINT_PATH'] != '') and (config['CHECKPOINT_PATH'] is not None):
        checkpoint = torch.load(os.path.join(rootdir, config['CHECKPOINT_PATH']))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        assert False, "no pretrained model!"

    # generate Non-Robust Dataset
    generate(trainloader, model, use_cuda)

def getImage(sample_image_path):
    im = Image.open(sample_image_path)
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        ])
    im = trans(im)
    return im

def select_one_sample_images(sample_inputs, sample_image_paths,
                             traindir, non_robust_dir,
                             targets, idx):
    sample_image_path_before = None
    sample_image_path_after = None
    # gurantee no override image
    while (1):
        sample_label = random.randint(0, 9)
        sample_path_before = os.path.join(traindir, str(sample_label))
        glob_files = os.path.join(sample_path_before, '*.png')
        files = glob.glob(glob_files)
        assert len(files) == 5000
        # choose the selected image randomly
        sample_image_path_before = random.choice(files)
        sample_image_name = sample_image_path_before.split('/')[-1]
        sample_image_dir_after = os.path.join(non_robust_dir, str(targets[idx].item())) 
        sample_image_path_after = os.path.join(sample_image_dir_after, sample_image_name)
        if (not os.path.exists(sample_image_path_after)) and (sample_image_path_after not in sample_image_paths): break

    assert os.path.exists(sample_image_path_before)
    sample_image = getImage(sample_image_path_before)

    if not os.path.isdir(
        os.path.join(non_robust_dir, str(targets[idx].item()))):
        os.makedirs(os.path.join(non_robust_dir, str(targets[idx].item())))
    sample_inputs[idx] = sample_image
    sample_image_paths.append(sample_image_path_after)
    return sample_inputs, sample_image_paths

def generate(trainloader, model, use_cuda):
    # switch to eval mode!
    model.eval()

    # original CIFAR-10 dataroot
    traindir = os.path.join(rootdir, 'data/cifar10_ori_images', 'train')
    non_robust_dir = os.path.join(rootdir, config['GENERATE']['SAVE_DIR'])

    assert os.path.isdir(traindir)
    if not os.path.isdir(non_robust_dir):
        os.makedirs(non_robust_dir)

    for k in range(config['DATA']['GENERATE_NUM']):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            start = time.time()
            print("[{}/{}]".format(str(batch_idx), str(len(trainloader))))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(async=True)

            b, c, h, w = inputs.shape
            sample_inputs = torch.zeros((b, c, h, w))
            sample_image_paths = []
            for idx, input in enumerate(inputs):
                sample_inputs, sample_image_paths = select_one_sample_images(sample_inputs, sample_image_paths,
                                                                             traindir, non_robust_dir,
                                                                             targets, idx)
            if use_cuda:
                sample_inputs = sample_inputs.cuda()

            # generate modified images
            sample_inputs = PGD(model, config['ATTACK']['STEP_SIZE'], config['ATTACK']['ITERATION'])(inputs, sample_inputs)
            assert len(sample_image_paths) == sample_inputs.shape[0]


            for idx, sample_image_path in enumerate(sample_image_paths):
                im_tensor = sample_inputs.cpu()[idx]
                im = ToPILImage(mode='RGB')(im_tensor)
                im.save(sample_image_path, quality=100)

            end = time.time()
            print('[{}/{}], time interval: {}s'.format(k, config['DATA']['GENERATE_NUM'], end-start))


if __name__ == '__main__':
    main()
