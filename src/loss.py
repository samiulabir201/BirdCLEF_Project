import torch
import torch.nn as nn

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        
        focal_loss = targets * self.alpha * (1 - probas) ** self.gamma * bce_loss + \
                     (1 - targets) * probas ** self.gamma * bce_loss
        return focal_loss.mean()
