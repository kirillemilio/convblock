''' Losses on PyTorch for various tasks '''

import torch


def dice_loss(y_true: 'Tensor', y_pred: 'Tensor') -> 'Tensor':
    """ Dice loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.

    Returns
    -------
    Tensor
        dice loss value.
    """
    smooth = 1e-7
    batch_size = y_true.size(0)
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    return -2 * torch.sum(y_true * y_pred) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


def tversky_loss(y_true: 'Tensor', y_pred: 'Tensor',
                 alpha=0.3, beta=0.7, smooth=1e-10) -> 'Tensor':
    """ Tversky loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.
    alpha : float
        alpha coefficient used in alpha * torch.sum(y_true * (1 - y_pred)) part.
    beta : float
        beta coefficient used in beta * torch.sum((1 - y_pred) * y_true) part.
    smooth : float
        small value used to avoid division by zero error.

    Returns
    -------
    Tensor
        tversky loss value.
    """
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    truepos = torch.sum(y_true * y_pred)
    fp_and_fn = (alpha * torch.sum((1 - y_true) * y_pred)
                 + beta * torch.sum((1 - y_pred) * y_true))

    return -(truepos + smooth) / (truepos + smooth + fp_and_fn)


def log_loss(y_true: 'Tensor', y_pred: 'Tensor') -> 'Tensor':
    """ Log loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.

    Returns
    -------
    Tensor
        log loss value.
    """
    smooth = 1e-7
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    return -torch.mean(y_true * torch.log(y_pred + smooth)
                       + (1 - y_true) * torch.log(1 - y_pred + smooth))


def dice_and_log_loss(y_true: 'Tensor', y_pred: 'Tensor', alpha=0.2) -> 'Tensor':
    """ Dice and log combination loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.
    alpha : float
        alpha coefficient used as (1 - alpha) * dice_loss + alpha * log_loss.

    Returns
    -------
    Tensor
        dice + log loss value.
    """
    return ((1 - alpha) * dice_loss(y_true, y_pred)
            + alpha * log_loss(y_true, y_pred))


class MultiBoxLoss(torch.nn.Module):

    def __init__(self, num_classes: int, ratio: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ratio = float(ratio)

    @classmethod
    def cross_entropy_loss(cls, x: 'Tensor(n, d)', y: 'Tensor(n)') -> 'Tensor(n)':
        """ Cross entropy loss averaging across all samples.

        Parameters
        ----------
        x : Tensor(n, d)
            predicted probabilities.
        y : Tensor(n)
            target labels.

        Returns
        -------
        Tensor(n)
            cross entropy loss, sized [N,].
        """
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1)).view(-1)

    @classmethod
    def hard_negative_mining(cls,
                             conf_loss: 'Tensor(n, d)',
                             pos: 'Tensor(n, d)', ratio: float) -> 'Tensor(n, d)':
        """ Return negative indices that is 3x the number as postive indices.

        Parameters
        ----------
        conf_loss : Tensor(n, d)
            confidence loss for bounding boxes.
        pos : Tensor(n, d)
            positive(matched) box indices, sized [N,8732].
        ratio : float
            ratio between negative and positive examples.

        Returns
        -------
        Tensor(n)
            tensor containing negative indices, sized [N,8732].
        """
        batch_size, num_boxes = pos.size()

        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss

        _, idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(1)  # [N,8732]

        num_pos = pos.long().sum(1).view(-1, 1)  # [N,1]
        num_neg = torch.clamp(3 * ratio * num_pos,
                              max=num_boxes - 1)  # [N,1]

        neg = rank < num_neg.expand_as(rank)  # [N,8732]
        return neg

    @classmethod
    def smooth_l1(cls, x: 'Tensor(n, d)', y: 'Tensor(n, d)') -> 'Tensor(1)':
        """ Compute smooth l1 loss. """
        z = torch.abs(x - y)
        return torch.sum(torch.where(z < 1.0, 0.5 * z * z, z - 0.5))

    def forward(self,
                loc_preds: 'Tensor(n, d, 4)',
                loc_targets: 'Tensor(n, d, 4)',
                conf_preds: 'Tensor(n, d, k)',
                conf_targets: 'Tensor(n, d)') -> 'Tensor(1)':
        """ Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).

        Parameters
        ----------
        loc_preds : Tensor(n, d, 4)
            predicted locations, sized [batch_size, 8732, 4].
        loc_targets : Tensor(n, d, 4)
            encoded target locations, sized [batch_size, 8732, 4].
        conf_preds : Tensor(n, d, #num_classes)
            predicted class confidences, sized [batch_size, 8732, num_classes].
        conf_targets : Tensor(n, d)
            encoded target classes, sized [batch_size, 8732].

        Returns
        -------
        Tensor(1)
            loss.
        """
        batch_size, num_boxes, _ = loc_preds.size()

        pos = conf_targets > 0  # [N,8732], pos means the box matched.
        num_matched_boxes = float(pos.data.long().sum())
        if num_matched_boxes == 0:
            return (Variable(torch.Tensor([0]), requires_grad=True),
                    Variable(torch.Tensor([0]), requires_grad=True),
                    Variable(torch.Tensor([0]), requires_grad=True))

        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)    # [N,8732,4]

        loc_loss = self.smooth_l1(loc_preds[pos_mask].view(-1, 4),
                                  loc_targets[pos_mask].view(-1, 4))

        conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes),
                                            conf_targets.view(-1)).view(batch_size, -1)

        neg = self.hard_negative_mining(conf_loss.detach(),
                                        pos, self.ratio)    # [N,8732]

        pos_mask = (
            pos
            .unsqueeze(2)
            .expand_as(conf_preds)
        )
        neg_mask = (
            neg
            .unsqueeze(2)
            .expand_as(conf_preds)
        )

        # [#pos+#neg,num_classes]
        preds = conf_preds[pos_mask + neg_mask > 0].view(-1, self.num_classes)
        targets = conf_targets[pos + neg > 0]

        conf_loss = F.cross_entropy(preds, targets, size_average=False)

        loc_loss /= num_matched_boxes
        conf_loss /= num_matched_boxes
        return loc_loss + conf_loss, conf_loss, loc_loss


class FocalLoss(torch.nn.Module):

    def __init__(self,
                 num_classes: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 2,
                 eps: float = 10e-7):
        super().__init__()
        self.num_classes = int(num_classes) + 1
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.eps = float(eps)

    @classmethod
    def smooth_l1(cls, x: 'Tensor(n, d)', y: 'Tensor(n, d)') -> 'Tensor(1)':
        """ Compute smooth l1 loss. """
        z = torch.abs(x - y)
        return torch.sum(torch.where(z < 1.0, 0.5 * z * z, z - 0.5))

    def forward_conf(self, conf_preds, conf_targets, pos_indices):
        num_matched_boxes = float(pos_indices.data.long().sum())
        pos_indices = pos_indices.nonzero().squeeze()
        targets = conf_targets.new(conf_preds.size()).float().zero_()
        targets[pos_indices, conf_targets[pos_indices]] = 1.0
        alpha_factor = self.alpha * targets.new_ones(targets.shape)

        alpha_factor = torch.where(torch.eq(targets, 1.),
                                   alpha_factor,
                                   1.0 - alpha_factor)

        focal_weight = torch.where(torch.eq(targets, 1.),
                                   1.0 - conf_preds,
                                   conf_preds)

        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = -(targets * torch.log(self.eps + conf_preds) +
                (1.0 - targets) * torch.log(1.0 - conf_preds + self.eps))
        return (focal_weight * bce).sum() / num_matched_boxes

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        pos_indices = conf_targets > 0
        num_matched_boxes = float(pos_indices.data.long().sum())
        if num_matched_boxes > 0:
            pos_mask = pos_indices.unsqueeze(2).expand_as(loc_preds)
            loc_loss = self.smooth_l1(loc_preds[pos_mask].view(-1, 4),
                                      loc_targets[pos_mask].view(-1, 4)) / num_matched_boxes
        else:
            loc_loss = 0

        conf_loss = self.forward_conf(conf_preds.view(-1, conf_preds.size(-1)),
                                      conf_targets.view(-1), pos_indices.view(-1))
        return self.beta * conf_loss + (1 - self.beta) * loc_loss, conf_loss, loc_loss
    

class IOULoss(torch.nn.Module):
    
    def __init__(self, mode='giou'):
        super().__init__()
        mode = mode.strip().lower()
        assert mode in ('iou', 'giou', 'diou', 'ciou')
        self.mode = mode
        
    @classmethod
    def iou(cls, x, y, mode='iou'):
        x_, y_ = x.unsqueeze(-2), y.unsqueeze(-3)

        tl = torch.max(x_[..., :2], y_[..., :2])
        br = torch.min(x_[..., 2:], y_[..., 2:])

        area_x = torch.prod(x_[..., 2:] - x_[..., :2], dim=-1).clamp(min=0.0)
        area_y = torch.prod(y_[..., 2:] - y_[..., 2:], dim=-1).clamp(min=0.0)
        
        area_inter = torch.prod(br - tl, dim=-1).clamp(min=0.0)
        area_union = area_x + area_y - area_inter
        
        iou = area_inter / area_union
        
        if mode == 'iou':
            return iou
        
        con_tl = torch.min(x_[..., :2], y_[..., :2])
        con_br = torch.max(x_[..., 2:], y_[..., 2:])
        area_con = torch.prod(con_br - con_tl, dim=-1).clamp(min=0.0)
        if mode == 'giou':
            return iou - (area_con - area_union) / area_con
        else:
            rho2 = ((x_[..., :2] + x_[..., 2:]) - (y_[..., :2] - y_[..., 2:])).pow(2.0).div(4).sum(dim=-1)
            c2 = (con_br - con_tl).pow(2.0).sum(dim=-1) + 1e-16

        if mode == 'diou':
            return iou - rho2 / c2
        else:
            q_x = torch.atan((x_[..., 2] - x_[..., 0]) / (x_[..., 3] - x_[..., 1]))
            q_y = torch.atan((y_[..., 2] - y_[..., 0]) / (y_[..., 3] - y_[..., 1]))
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(q_x) - torch.atan(q_y), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            return iou - (rho2 / c2 + v * alpha)
    
    def forward(self, pred, target):
        return 1.0 - self.iou(pred, target, mode=self.mode)
