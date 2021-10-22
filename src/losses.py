import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


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


class AngularMarginLoss(MetricLearningLoss):
    """
    Generic angular margin loss definition
    (see https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch)

    "ElasticFace: Elastic Margin Loss for Deep Face Recognition",
    Boutros et al., https://arxiv.org/abs/2109.09416v2
    """

    def __init__(
        self, embedding_size, n_classes, scale=None, m1=1, m2=0, m3=0, eps=1e-6
    ):
        super(AngularMarginLoss, self).__init__(embedding_size, n_classes)
        self.fc = nn.Linear(embedding_size, n_classes, bias=False)
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Compute ArcFace loss for inputs of shape [B, E] and
        targets of size [B]

        B: batch size
        E: embedding size
        """
        # Normalize weights
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)

        # Normalize inputs
        inputs_norms = torch.norm(inputs, p=2, dim=1)
        normalized_inputs = inputs / inputs_norms.unsqueeze(-1).repeat(
            1, inputs.size(1)
        )

        # Set scale
        scales = (
            torch.tensor([self.scale], device=inputs.device).repeat(inputs.size(0))
            if self.scale is not None
            else inputs_norms
        )

        # Cosine similarity is given by a simple dot product,
        # given that we normalized both weights and inputs
        cosines = self.fc(normalized_inputs).clamp(-1, 1)
        preds = torch.argmax(cosines, dim=1)

        # Recover angles from cosines computed
        # from the previous linear layer
        angles = torch.arccos(cosines)

        # Compute loss numerator by converting angles back to cosines,
        # after adding penalties, as if they were the output of the
        # last linear layer
        numerator = scales.unsqueeze(-1) * (
            torch.cos(self.m1 * angles + self.m2) - self.m3
        )
        numerator = torch.diagonal(numerator.transpose(0, 1)[targets])

        # Compute loss denominator
        excluded = torch.cat(
            [
                scales[i]
                * torch.cat((cosines[i, :y], cosines[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(targets)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excluded), dim=1)

        # Compute cross-entropy loss
        loss = -torch.mean(numerator - torch.log(denominator + self.eps))

        return inputs, preds, loss


class SphereFaceLoss(AngularMarginLoss):
    """
    Compute the multiplicative angular margin loss

    "SphereFace: Deep Hypersphere Embedding for Face Recognition",
    Liu et al., https://arxiv.org/abs/1704.08063
    """

    def __init__(self, embedding_size, n_classes, scale=None, margin=3, eps=1e-6):
        assert margin > 1, "Margin out of bounds"
        super(SphereFaceLoss, self).__init__(
            embedding_size, n_classes, scale=scale, m1=margin, eps=eps
        )


class CosFaceLoss(AngularMarginLoss):
    """
    Compute the additive cosine margin loss

    "CosFace: Large Margin Cosine Loss for Deep Face Recognition",
    Wang et al., https://arxiv.org/abs/1801.09414
    """

    def __init__(self, embedding_size, n_classes, scale=64, margin=0.2, eps=1e-6):
        assert margin > 0 and margin < 1 - np.cos(np.pi / 4), "Margin out of bounds"
        super(CosFaceLoss, self).__init__(
            embedding_size, n_classes, scale=scale, m3=margin, eps=eps
        )


class ArcFaceLoss(AngularMarginLoss):
    """
    Compute the additive angular margin loss

    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
    Deng et al., https://arxiv.org/abs/1801.07698
    """

    def __init__(self, embedding_size, n_classes, scale=64, margin=0.5, eps=1e-6):
        assert margin > 0 and margin < 1, "Margin out of bounds"
        super(ArcFaceLoss, self).__init__(
            embedding_size, n_classes, scale=scale, m2=margin, eps=eps
        )


LOSSES = {
    "ce": CELoss,
    "sphere": SphereFaceLoss,
    "cos": CosFaceLoss,
    "arc": ArcFaceLoss,
}
