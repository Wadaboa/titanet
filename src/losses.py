import torch
import torch.nn.functional as F
import torch.nn as nn


class MetricLearningLoss(nn.Module):
    """
    Generic loss function to be used in a metric learning setting
    """

    def __init__(self, embedding_size, n_classes, *args, **kwargs):
        super(MetricLearningLoss, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes

    def forward(self, inputs, targets):
        raise NotImplementedError()


class CELoss(MetricLearningLoss):
    """
    Cross-entropy loss with the addition of a linear layer
    to map inputs to the target number of classes
    """

    def __init__(self, embedding_size, n_classes):
        super(CELoss, self).__init__(embedding_size, n_classes)
        self.fc = nn.Linear(embedding_size, n_classes)

    def forward(self, inputs, targets):
        """
        Compute cross-entropy loss for inputs of shape
        [B, E] and targets of size [B]

        B: batch size
        E: embedding size
        """
        logits = self.fc(inputs)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)
        return inputs, preds, loss


class ArcFaceLoss(MetricLearningLoss):
    """
    Compute the additive angular margin loss

    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
    Deng et al., https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self, embedding_size, n_classes, normalize_input=True, scale=64, margin=0.5
    ):
        super(ArcFaceLoss, self).__init__(embedding_size, n_classes)
        self.fc = nn.Linear(embedding_size, n_classes, bias=False)
        self.normalize_input = normalize_input
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        """
        Compute ArcFace loss for inputs of shape [B, E] and
        targets of size [B]

        B: batch size
        E: embedding size
        """
        # Normalize weights
        for parameter in self.fc.parameters():
            parameter = F.normalize(parameter, p=2, dim=1)

        # Normalize inputs
        if self.normalize_input:
            inputs = F.normalize(inputs, p=2, dim=1)

        # Cosine similarity is given by a simple dot product,
        # given that we normalized both weights and inputs
        cosines = self.fc(inputs)

        # Recover angles from cosines computed
        # from the previous linear layer
        angles = torch.arccos(cosines)

        # Compute logits by converting angles back to cosines,
        # after adding penalties, as if they were the output
        # of the last linear layer
        logits = self.scale * torch.cos(angles + self.margin)
        preds = torch.argmax(logits, dim=1)

        # Rely on standard cross-entropy using modified logits
        loss = F.cross_entropy(logits, targets)

        return inputs, preds, loss


LOSSES = {"ce": CELoss, "aam": ArcFaceLoss}
