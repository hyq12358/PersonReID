import torch
import torch.nn.functional as F
def accuracy(y_pred, y_true):
    """
    Calculate accuracy.

    Args:
    -----------------------------------
    * y_pred: Tensor of shape (N, K) or (N,) predicted prob.
    * y_true: Tensor of shape (N,), true label

    Returns:
    * acc: scalar, accuracy
    """
    if y_pred.dim() == 1:
        acc = (y_pred==y_true)
    else:    
        acc = (torch.argmax(y_pred, dim=-1)==y_true)
    return torch.mean(acc.to(dtype=torch.float32)).item()


def top1_accuracy(query, query_features, gallary, gallary_features, normalize=True):
    """
    Calculate top1 accuracy.

    Args:
    -----------------------------------
    * query: Tensor of shape (N,), query label
    * query_features: Tensor of shape(N, K)
    * gallary: Tensor of shape (M,), gallary label
    * gallary_features: Tensor of shape(N, K)

    Returns:
    * top1_acc: scalar, accuracy
    """
    if normalize:
        query_features = F.normalize(query_features, p=2, dim=-1)
        gallary_features = F.normalize(gallary_features, p=2, dim=-1)
    similarity = torch.mm(query_features, gallary_features.t())
    query_pred = gallary[torch.argmax(similarity, dim=-1)]
    top1_acc = accuracy(query, query_pred)
    return top1_acc

    