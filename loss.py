import torch.nn as nn
import torch
class DiceLoss(nn.Module):
    def forward(self, predictions, targets, smooth=1):
        predictions = torch.sigmoid(predictions)  # Apply sigmoid to get probabilities
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        return 1 - dice

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Includes sigmoid inside
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return bce_loss + dice_loss
