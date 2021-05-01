'''
Training script for modified CIFAR-10.
Copyright (c) Peidong Liu, 2020
'''
import argparse
import os
import yaml
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as ours_models_cifar
from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy, mkdir_p


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config', default='', type=str)
args = parser.parse_args()

config_path = args.config
assert config_path != '', 'you haven`t assign any config file yet!'
config = yaml.load(open(config_path))
task_name = config['TASK_NAME']
root_dir = config['ROOT_DIR']
save_dir = os.path.join(root_dir, config['SAVE_DIR'])
if config['MODEL']['TORCHVISION']:
    models = torchvision.models
else:
    models = ours_models_cifar
# Validate dataset
dataset = config['DATA']['DATASET']
writer = SummaryWriter(os.path.join(save_dir, task_name))

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
use_cuda = torch.cuda.is_available()

# Random seed
seed = config['SEED']
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

best_acc = 0  # best test accuracy
def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    checkpoint_dir = os.path.join(save_dir, task_name, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        mkdir_p(checkpoint_dir)

    # Data
    print('==> Preparing dataset %s' % dataset, flush=True)
    num_classes = 10
    dataloader = datasets.CIFAR10

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.25, .25, .25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])
    if not config['DATA']['AUGMENTATION']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            ])
    traindir = os.path.join(root_dir, config['DATA']['TRAIN_DATAPATH'], 'train')
    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transform=transform_train),
        batch_size=config['TRAIN']['BATCH_SIZE'], shuffle=True,
        num_workers=config['DATA']['NUM_WORKERS'], pin_memory=True)

    testset = dataloader(root=os.path.join(root_dir, config['DATA']['VAL_DATAPATH']),
                         train=False, download=False,
                         transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['VALID'][
        'BATCH_SIZE'], shuffle=False, num_workers=config['DATA'][
        'NUM_WORKERS'])


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

    criterion = nn.CrossEntropyLoss().cuda()
    optim_type = config['SOLVER']['TYPE']
    optimizer = optim.__dict__[optim_type](model.parameters(), lr=config['SOLVER']['LR'], momentum=config['SOLVER']['MOMENTUM'], weight_decay=config['SOLVER']['WEIGHT_DECAY'])
    
    if config['SOLVER']['LR_POLICY'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.__dict__[config['SOLVER']['LR_POLICY']](optimizer, step_size=config['SOLVER']['LR_STEP'])
    elif config['SOLVER']['LR_POLICY'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.__dict__[config['SOLVER']['LR_POLICY']](optimizer, milestones=config['SOLVER']['MULTI_LR_STEPS'])
    elif config['SOLVER']['LR_POLICY'] == 'CosineAnnealingLR':
        max_iters = config['SOLVER']['MAX_EPOCH'] * len(trainloader)
        scheduler = torch.optim.lr_scheduler.__dict__[config['SOLVER']['LR_POLICY']](optimizer, T_max=max_iters)
    else:
        raise RuntimeError('not supported lr scheduler')

    # Resume
    if config['RESUME']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..', flush=True)
        resume_path = os.path.join(save_dir, task_name, 'checkpoints/checkpoint.pth')
        assert os.path.isfile(resume_path), 'Error: no checkpoint found!'
        checkpoint = torch.load(resume_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if config['ONLY_EVALUATE']:
        print('\nEvaluation only', flush=True)
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc), flush=True)
        return

    # Train and val
    for epoch in range(start_epoch, config['SOLVER']['MAX_EPOCH']):
        cur_lr = scheduler.get_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, config['SOLVER']['MAX_EPOCH'], cur_lr), flush=True)
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, scheduler)
        if (epoch + 1) % config['VALID']['EPOCH'] == 0:
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=checkpoint_dir)
            writer.add_scalar('epoch-metric/test loss', test_loss, epoch + 1)
            writer.add_scalar('epoch-metric/train loss', train_loss, epoch + 1)
            writer.add_scalar('epoch-metric/test accuracy', test_acc, epoch + 1)
            writer.add_scalar('epoch-metric/train accuracy', train_acc, epoch + 1)
            writer.add_scalar('valid-best-metric/test best accuracy', best_acc, epoch + 1)


    print('best acc: ', best_acc, flush=True)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, scheduler):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        scheduler.step()
        cur_lr = scheduler.get_lr()[0]
        writer.add_scalar('lr', cur_lr, batch_idx+epoch*len(trainloader))
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        try:
            outputs = model(inputs, with_latent=False, fake_relu=True)
        except:
            try:
                outputs = model(inputs, single=True)
            except:
                outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        with torch.no_grad():
            if batch_idx % config['RECORD']['PRINT_FREQUENCY'] == 0:
                print('[TASK_NAME: {}] ({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | lr: {lr: .4f}'.format(
                        config['TASK_NAME'],
                        batch=batch_idx,
                        size=len(trainloader),
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        lr=cur_lr
                        ), flush=True)
                writer.add_scalar('train-batch/train loss', loss.item(), batch_idx)
                writer.add_scalar('train-batch/train accuracy', prec1.item(), batch_idx)

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    with torch.no_grad():
        global best_acc

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            try:
                outputs = model(inputs, with_latent=False, fake_relu=True)
            except:
                try:
                    outputs = model(inputs, single=True)
                except:
                    outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress
            if batch_idx % config['RECORD']['PRINT_FREQUENCY'] == 0:
                print('[TASK_NAME: {}] ({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        config['TASK_NAME'],
                        batch=batch_idx,
                        size=len(testloader),
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ), flush=True)
        return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))

if __name__ == '__main__':
    main()
