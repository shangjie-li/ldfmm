import torch


def focal_loss(input, target, gamma=2.0):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * neg_weights

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss.mean()
