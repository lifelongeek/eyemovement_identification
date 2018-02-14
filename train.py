import argparse
import errno
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from loader import FeatLoader, FeatDataset, BucketingSampler

from model import StackedBRNN, supported_rnns
from utils import AverageMeter, _get_variable_nograd, _get_variable_volatile, _get_variable

import pdb


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description='Eye movement')
parser.add_argument('--expnum', type=int, default=0)


parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_shuffle.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/valid_shuffle.csv')
parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--max_value', default=400.0, type=float, help='maximum absolute value for feature normalization')


parser.add_argument('--rnn_hidden', default=128, type=int, help='Hidden size of RNNs')
parser.add_argument('--rnn_layers', default=2, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--dropout', default=0)

parser.add_argument('--fc_hidden', default=128)
parser.add_argument('--fc_layers', default=2)
parser.add_argument('--nClass', default=57)

parser.add_argument('--print_every', type = int, default=100)

parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
parser.add_argument('--gpu', default=0, type=int, help='-1 : cpu')

parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float, help='initial learning rate')
parser.add_argument('--optim', default='adam', help='adam|sgd')

parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='', help='Location to save best validation model')  # set up this in code

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'

    save_folder = args.save_folder

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    loss_train, acc_valid = torch.Tensor(args.epochs), torch.Tensor(args.epochs)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = StackedBRNN(input_size=2, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, rnn_type=supported_rnns[args.rnn_type],
                        dropout=args.dropout,
                        fc_hidden = args.fc_hidden, fc_layers=args.fc_layers, nClass = args.nClass)
    #model.apply(weights_init) # use default initialization
    print(model)
    if args.gpu >=0:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False)

    avg_loss, start_epoch, start_iter = 0, 0, 0


    parameters = model.parameters()

    # Optimizer
    if(args.optim == 'adam'):
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    elif(args.optim == 'sgd'):
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)

    # Data loader
    train_dataset = FeatDataset(manifest_filepath=args.train_manifest, maxval=args.max_value)
    test_dataset = FeatDataset(manifest_filepath=args.val_manifest, maxval=args.max_value)
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_sampler.shuffle()
    train_loader = FeatLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = FeatLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()



    # Save model file for error check
    file_path = '%s/%d.pth.tar' % (save_folder, args.expnum)  # always overwrite recent epoch's model
    torch.save(model.state_dict(), file_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        losses.reset()

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            # measure data loading time
            data_time.update(time.time() - end)

            # load data
            input, target = data
            input = _get_variable_nograd(input, cuda=True)
            target = _get_variable_nograd(target, cuda=True)

            # Forward
            output = model(input)
            #print(target)
            loss = criterion(output,target)
            loss = loss / input.size(0)  # average the loss by minibatch
            losses.update(loss.data[0], input.size(0))

            # Backprop
            model.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_every == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        (epoch + 1), (i + 1), len(train_sampler),
                        batch_time=batch_time,
                        data_time=data_time, loss=losses))

            del loss
            del output


        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        losses.reset()
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # load data
            input, target = data
            input = _get_variable_volatile(input, cuda=True)
            target = _get_variable_volatile(target, cuda=True)

            # Forward
            output = model(input)
            loss = criterion(output,target)
            loss = loss / input.size(0)  # average the loss by minibatch
            losses.update(loss.data[0], input.size(0))

            # TODO : measure accuracy

        valid_accuracy = 0 # TODO



        print('Validation Summary Epoch: [{0}]\t'
              'Average loss {loss:.3f}\t'
              'Average acc {acc:.3f}\t'.format(
            epoch + 1, loss=losses.avg, acc=valid_accuracy))


        file_path = '%s/%d.pth.tar' % (save_folder, args.expnum)  # always overwrite recent epoch's model
        torch.save(model.state_dict(), file_path)

        print("Shuffling batches...")
        train_sampler.shuffle()
