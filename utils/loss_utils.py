import torch


def focal_loss(inputs, targets, gamma=2.0):
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)
    pos_loss = torch.log(inputs) * torch.pow(1 - inputs, gamma) * pos_inds
    neg_loss = torch.log(1 - inputs) * torch.pow(inputs, gamma) * neg_inds * neg_weights

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss.mean()
