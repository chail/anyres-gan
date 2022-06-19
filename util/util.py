import torch
import urllib

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
