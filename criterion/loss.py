import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class TripletLoss(nn.Module):
    """
    Triplet Loss. see more at `https://machinelearning.wtf/terms/triplet-loss/`
    """
    def __init__(self, alpha=1., reduction='mean'):
        super().__init__()
        self.alpha = alpha
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            self.reduction = reduction

    def forward(self, anchor, postive, negative, normalize=True):
        if normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            postive = F.normalize(postive, p=2, dim=-1)
            negative = F.normalize(negative, p=2, dim=-1)
        return self.reduction(
            F.relu(
                torch.norm(anchor-postive, dim=-1)
                + self.alpha
                - torch.norm(anchor-negative, dim=-1)
            )
        )

