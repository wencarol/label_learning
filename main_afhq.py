
from __future__ import print_function

import argparse
import hashlib
import os
import random
import shutil
import time
import torchvision.models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
# from progress.bar import Bar

import datasets.registry
import models.registry
# import utils.mix_utils
from criterions import cpu_risk
from utils.logger import Logger
from utils.misc import mkdir_p

parser = argparse.ArgumentParser(description='CIFAR10/100-Q Training')
# Datasets
parser.add_argument('--dataset', default='afhq', type=str)
parser.add_argument('--dataset_path', default='/root/paddlejob/workspace/env_run/expr/local_editing/afhq/synthesized_image/', type=str)
parser.add_argument('--val_path', default='/root/paddlejob/workspace/env_run/data/afhq/raw_images/val/images/', type=str)
parser.add_argument('--test_path', default='/root/paddlejob/workspace/env_run/data/afhq/raw_images/test/images/', type=str)
parser.add_argument('--mix_mode', default='mixup', type=str, help='choose which mode for image mixing \
                    in cifar-q generation', choices=['mixup', 'patchmix+'])
parser.add_argument('--mixing_complexity', default=2, type=int, help='number of images to mix')
parser.add_argument('--trail_num', default=100, type=int, help='number of trails to when sampling from multinomial distribution.')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset_seed', type=int, default=12345, help='random seed for transforming dataset')

# Optimization options
parser.add_argument('--loss_type', type=str, default='js', help='The loss function used in pu estimator.',)
parser.add_argument('--pi_1', type=float, default=0.1, help='The class prior probability for positive risk.')
parser.add_argument('--pi_2', type=float, default=0.6, help='The class prior probability.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

# Checkpoints
parser.add_argument('--checkpoint_dir', '-c', default='results/pu_loss', type=str, metavar='PATH',
                    help='root dir to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--model_name', default='imagenet_resnet50',)
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--drop_rate', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--block_name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck.\
                        default: Basicblock for cifar10/cifar100')
parser.add_argument('--width_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

# Evaluate
parser.add_argument('-e', '--evaluate_only', action='store_true',
                    help='Evaluation mode')                  

# Miscs
parser.add_argument('--training_seed', type=int, default=1, help='random seed for training')
parser.add_argument('--display_location', '-d', action='store_true',
                    help='show the output path of stored checkpoint given the hyperparameters.')
parser.add_argument('--quiet', action='store_true',
                    help='suppress output of training and testing status.')

# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device_str = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
print(torch.cuda.device_count())
device = torch.device(device_str)


# Random seed
if args.training_seed is not None:
    torch.manual_seed(args.training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.training_seed)
    random.seed(args.training_seed)
if device_str == 'cuda':
    torch.cuda.manual_seed_all(args.training_seed)


def train(trainloader, model, num_classes, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    example_count = torch.tensor(0.0).to(device)
    total_loss = torch.tensor(0.0).to(device)
    total_correct = torch.tensor(0.0).to(device)

    if not args.quiet:
        time_of_last_epoch = time.time()
    # if not args.quiet: bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        targets = targets.to(device)
        _targets = nn.functional.one_hot(targets, num_classes).float() if args.loss_type == 'bce' else targets
        inputs = inputs.to(device)

        # compute output
        outputs = model(inputs)

        # compute loss
        labels_size = torch.tensor(len(targets), device=device)
        example_count += labels_size
        loss = criterion(outputs, _targets)
        total_loss += loss * labels_size
        correct = torch.sum(torch.eq(targets, outputs.argmax(dim=1)))
        total_correct += correct

        optimizer.zero_grad()

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

    total_loss = total_loss.cpu().item()
    total_correct = total_correct.cpu().item()
    example_count = example_count.cpu().item()

    if not args.quiet:
        print('Time: {total:.2f}s | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
              total=time.time() - time_of_last_epoch,
              loss=total_loss / example_count,
              top1=100 * total_correct / example_count,))
        time_of_last_epoch = time.time()

    return (total_loss / example_count, total_correct / example_count)


def test(testloader, model, num_classes, criterion):
    global best_acc

    example_count = torch.tensor(0.0).to(device)
    total_loss = torch.tensor(0.0).to(device)
    total_correct = torch.tensor(0.0).to(device)

    if not args.quiet:
        time_of_last_epoch = time.time()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)

            labels_size = torch.tensor(len(targets), device=device)
            example_count += labels_size

            loss = criterion(outputs, targets)
            total_loss += loss * labels_size
            correct = torch.sum(torch.eq(targets, outputs.argmax(dim=1)))
            total_correct += correct

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()       

    if not args.quiet:
        print('Time: {total:.2f}s | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
              total=time.time() - time_of_last_epoch,
              loss=total_loss / example_count,
              top1=100 * total_correct / example_count,))
        time_of_last_epoch = time.time()
    
    return (total_loss / example_count, total_correct / example_count)

def parse_config(args):

    # Set checkpoint subdir path
    args_dict = {f: getattr(args, f) for f in vars(args)}
    dataset_prefix_list = ['dataset', 'mix_mode', 'mixing_complexity'] 

    checkpoint_prefix_list = dataset_prefix_list
    checkpoint_prefix_name = ''
    for s in checkpoint_prefix_list:
        prefix_s = str(getattr(args, s)) + '_'
        checkpoint_prefix_name += prefix_s
    unrelated_hyparams_list = ['checkpoint_dir', 'resume', 'evaluate_only', 'display_location', 'gpu',
                               'dataset_path', 'test_batch', 'quiet']
    for s in unrelated_hyparams_list:
        args_dict.pop(s)

    # Set model saved path and args dictionary
    if args.resume:
        checkpoint_subdir = os.path.dirname(args.resume)
    else:
        if not os.path.isdir(args.checkpoint_dir):
            mkdir_p(args.checkpoint_dir)
        hparams_strs = [str(args_dict[k]) for k in sorted(args_dict)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        checkpoint_subdir = os.path.join(args.checkpoint_dir, checkpoint_prefix_name + hash_str)
        if not os.path.isdir(checkpoint_subdir) and not args.resume:
            mkdir_p(checkpoint_subdir)
        
    return checkpoint_subdir, args_dict


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


best_acc = 0  # best test accuracy
def main():

    # Set model saved path and args dictionary
    checkpoint_subdir, args_dict = parse_config(args)

    # Display experimental details
    print('===> Hyperparameters:')
    for k,v in args_dict.items():
        print('%s:    %s'%(k, v))
    print('==> Output location \n    %s' % checkpoint_subdir)
    if args.display_location: return

    # Data used during training or evaluating
    if not args.evaluate_only:
        print('==> Preparing dataset %s' % args.dataset)
        trainset, num_classes = datasets.registry.get(data_name=args.dataset, data_path=args.dataset_path, train=True,
                                                      mixing_complexity=args.mixing_complexity, 
                                                      mix_mode=args.mix_mode, trail_num=args.trail_num,
                                                      seed=args.dataset_seed)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    
    print('==> Preparing original test dataset %s' % args.dataset)
    testset, num_classes = datasets.registry.get(args.dataset, args.dataset_path, train=False, seed=args.dataset_seed, test_root=args.test_path,
                                                      val_root=args.val_path)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> Creating model '{}'".format(args.model_name))
    model = models.registry.get(args.model_name, num_classes=num_classes, depth=args.depth,
                                width_factor=args.width_factor, drop_rate=args.drop_rate)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    if args.loss_type == 'ce':
        train_criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'bce':
        train_criterion = nn.BCEWithLogitsLoss()
    else:
        train_criterion = cpu_risk.Loss(loss_type=args.loss_type, pi_1=args.pi_1, pi_2=args.pi_2,
                                        device=device, num_classes=num_classes)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if os.path.exists(os.path.join(checkpoint_subdir, 'checkpoint.pth')) and not args.resume:
        args.resume = os.path.join(checkpoint_subdir, 'checkpoint.pth')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        loaded_checkpoint = torch.load(args.resume)
        best_acc = loaded_checkpoint['best_acc']
        test_acc = loaded_checkpoint['acc']
        start_epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_subdir, 'log.txt'), resume=True)

        # Evaluate only
        if args.evaluate_only:
            print('Evaluation only')
            test_loss, test_acc = test(testloader, model, num_classes, criterion)
            print('Ground-truth Test Dataset: %s, Loss:  %.8f, Acc:  %.4f' % (args.dataset, test_loss, 100 * test_acc))
            logger.write_dict({'Ground-truth Test Dataset': args.dataset, 'Loss': test_loss, 'Acc': test_acc}) 
            logger.close()
            return
    else:
        if args.evaluate_only:
            raise ValueError("Checkpoint path must be assigned if you want to evaluate certain model!")
        logger = Logger(os.path.join(checkpoint_subdir, 'log.txt'))
        # logger.write_dict(args_dict)
        logger.set_names(['Epochs', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.',])
        hparams_logger = Logger(os.path.join(checkpoint_subdir, 'hyperparameters.txt'))
        hparams_logger.write_dict(args_dict)
        hparams_logger.close()

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if not args.quiet: print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, num_classes, train_criterion, optimizer, epoch)        # soft_train_loss, soft_train_acc = test(trainloader, model, num_classes, criterion, True)
        test_loss, test_acc = test(testloader, model, num_classes, criterion)
        logger.append([epoch, train_loss, test_loss, train_acc, test_acc,])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'acc': test_acc,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        }, is_best, checkpoint=checkpoint_subdir)
    logger.write_dict({'Best acc of original test data': best_acc})
    print('For original test data, last epoch acc: %.4f, best acc: %.4f'%(100 * test_acc ,100 * best_acc))

    logger.close()
    return 

if __name__ == '__main__':
    main()
