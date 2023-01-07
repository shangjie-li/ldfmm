import torch.optim as optim


def build_optimizer(cfg, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [
        {'params': biases, 'weight_decay': 0},
        {'params': weights, 'weight_decay': cfg['weight_decay']}
    ]

    if cfg['type'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=0.9)
    elif cfg['type'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg['lr'])
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg['type'])

    return optimizer
