# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    args = parser.add_argument_group('Training', 'model training')
    args.add_argument('--network', type=str,
                      help='the neural network to use')
    args.add_argument('--dataset', type=str, default='NTU',
                      help='select dataset to evlulate')
    args.add_argument('--start-epoch', default=0, type=int,
                      help='manual epoch number (useful on restarts)')
    args.add_argument('--max-epoches', type=int, default=150,
                      help='max number of epochs to run')
    args.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
    args.add_argument('--lr-factor', type=float, default=0.1,
                      help='the ratio to reduce lr on each step')
    args.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                      help='weight decay (default: 1e-4)')
    args.add_argument('--print-freq', '-p', type=int, default=10,
                      help='print frequency (default: 10)')
    args.add_argument('-b', '--batch-size', type=int, default=256,
                      help='mini-batch size (default: 256)')
    args.add_argument('--num-classes', type=int, default=60,
                      help='the number of classes')
    args.add_argument('--case', type=int, default=0,
                      help='select which case')
    args.add_argument('--train', type=int, default=1,
                      help='train or test')
    args.add_argument('--workers', type=int, default=2,
                      help='number of data loading workers (default: 2)')
    args.add_argument('--monitor', type=str, default='val_acc',
                      help='quantity to monitor (default: val_acc)')
    args.add_argument('--seg', type=int, default=20,
                      help='number of segmentation')
    return args

