import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import errno
import os
import os.path as osp
import shutil
import argparse

from model.lvt_cls import lvt_same_flops, lvt_upsample, lvt_same_params
from model.crossvit import crossvit_tiny_224

from utils.mixup import mixup_train, manifold_mixup_train
from utils.cutmix import cutmix_train
from utils.util import train, test

model_options = ['lvt_finetune', 'lvt_params', 'lvt_flops', 'crossvit']
augment_options = ['baseline', 'mixup', 'manimix', 'cutmix', 'mixtoken']

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 16)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', type=str, default='lvt_finetune',
                    choices=model_options)
parser.add_argument('--dataaug', type=str, default='baseline',
                    choices=augment_options)

args = parser.parse_args()

train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

train_dataset = CIFAR100(root='/root/cvmid-ass1/cifar100', train=True, download=False, transform=train_transform)
valid_dataset = CIFAR100(root='/root/cvmid-ass1/cifar100', train=False, download=False, transform=val_transform)

Batch_size = args.batch_size
train_loader = DataLoader(train_dataset,
                              batch_size=Batch_size,
                              shuffle=True,
                              num_workers=2)
valid_loader = DataLoader(valid_dataset,
                            batch_size=Batch_size,
                            num_workers=2)

print('==>Building model..')

if args.model == 'lvt_finetune':
    model = lvt_upsample()
if args.model == 'lvt_params':
    model = lvt_same_params()
if args.model == 'lvt_flops':
    model = lvt_same_flops()
if args.model == 'crossvit':
    model = crossvit_tiny_224(pretrained=True)
    embed_dim = [96, 192]
    num_branches = 2
    num_classes = 100
    model.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(num_branches)])

if args.dataaug == 'baseline':
    train_pro = train
if args.dataaug == 'mixup':
    train_pro = mixup_train
if args.dataaug == 'cutmix':
    train_pro = cutmix_train
if args.dataaug == 'manimix':
    train_pro = manifold_mixup_train
if args.dataaug == 'mixtoken':
    train_pro = train
    model.mixtoken = True

model = model.cuda()
torch.backends.cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile('./best_model_cvt.pth'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./best_model_cvt.pth')
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_acc = 0
    start_epoch = 0


path = './log/trans'
writer = SummaryWriter(path)


def mkdir_if_missing(directory):
    #创建文件夹，如果这个文件夹不存在的话
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best=False, fpath=''):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

num_epochs = args.epochs

train_loader = train_loader
test_loader = valid_loader
optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=start_epoch-1)
criterion = nn.CrossEntropyLoss()
print('==>Start training..')
for epoch in range(start_epoch, num_epochs):
    _, train_log = train_pro(train_loader, model, criterion, optimizer, epoch)
    test_loss, test_acc1, test_log = test(test_loader, model, criterion)
    train_loss, train_acc1, _ = test(train_loader, model, criterion)
    writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('top1 acc', {'train': train_acc1, 'test': test_acc1}, epoch)
    # writer.add_scalars('top5 acc', {'train': train_acc5, 'test': test_acc5}, epoch)
    scheduler.step()
    log = train_log + test_log
    print(log)
    is_best = test_acc1 > best_acc
    best_acc = max(test_acc1, best_acc)
    if is_best:
        save_checkpoint({'epoch':epoch,
        'state_dict':model.state_dict(),
        'acc': best_acc,
        }, False, 'best_model_transmix.pth')
