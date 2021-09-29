import torch.nn.functional as F
import torch

def entropy_loss(predict, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):

    if class_level_weight is None:
        class_level_weight = 1.0
    if instance_level_weight is None:
        instance_level_weight = 1.0
    predict = F.softmax(predict, dim=-1)
    entropy = -predict * torch.log(predict + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / predict.shape[0]




