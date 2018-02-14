import pdb
from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_variable(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda())
    else:
        out = Variable(inputs)
    return out


def _get_variable_volatile(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda(), volatile=True)
    else:
        out = Variable(inputs, volatile=True)
    return out


def _get_variable_nograd(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda(), requires_grad=False)
    else:
        out = Variable(inputs, requires_grad=False)
    return out