import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class MetricLearningLoss(nn.Module):
    """
    Generic loss function to be used in a metric learning setting
    """

    def __init__(self, embedding_size, n_classes, device="cpu", *args, **kwargs):
        super(MetricLearningLoss, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.device = device

    def forward(self, inputs, targets):
        raise NotImplementedError()


class CELoss(MetricLearningLoss):
    """
    Cross-entropy loss with the addition of a linear layer
    to map inputs to the target number of classes
    """

    def __init__(self, embedding_size, n_classes, device="cpu"):
        super(CELoss, self).__init__(embedding_size, n_classes, device=device)
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
        inputs = F.normalize(inputs, p=2, dim=1)
        return inputs, preds, loss


class AngularMarginLoss(MetricLearningLoss):
    """
    Generic angular margin loss definition
    (see https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch)

    "ElasticFace: Elastic Margin Loss for Deep Face Recognition",
    Boutros et al., https://arxiv.org/abs/2109.09416v2
    """

    def __init__(
        self,
        embedding_size,
        n_classes,
        device="cpu",
        scale=None,
        m1=1,
        m2=0,
        m3=0,
        eps=1e-6,
    ):
        super(AngularMarginLoss, self).__init__(
            embedding_size, n_classes, device=device
        )
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

        return normalized_inputs, preds, loss


class SphereFaceLoss(AngularMarginLoss):
    """
    Compute the multiplicative angular margin loss

    "SphereFace: Deep Hypersphere Embedding for Face Recognition",
    Liu et al., https://arxiv.org/abs/1704.08063
    """

    def __init__(
        self, embedding_size, n_classes, device="cpu", scale=None, margin=3, eps=1e-6
    ):
        assert margin > 1, "Margin out of bounds"
        super(SphereFaceLoss, self).__init__(
            embedding_size, n_classes, device=device, scale=scale, m1=margin, eps=eps
        )


class CosFaceLoss(AngularMarginLoss):
    """
    Compute the additive cosine margin loss

    "CosFace: Large Margin Cosine Loss for Deep Face Recognition",
    Wang et al., https://arxiv.org/abs/1801.09414
    """

    def __init__(
        self, embedding_size, n_classes, device="cpu", scale=64, margin=0.2, eps=1e-6
    ):
        assert margin > 0 and margin < 1 - np.cos(np.pi / 4), "Margin out of bounds"
        super(CosFaceLoss, self).__init__(
            embedding_size, n_classes, device=device, scale=scale, m3=margin, eps=eps
        )


class ArcFaceLoss(AngularMarginLoss):
    """
    Compute the additive angular margin loss

    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
    Deng et al., https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self, embedding_size, n_classes, device="cpu", scale=64, margin=0.5, eps=1e-6
    ):
        assert margin > 0 and margin < 1, "Margin out of bounds"
        super(ArcFaceLoss, self).__init__(
            embedding_size, n_classes, device=device, scale=scale, m2=margin, eps=eps
        )


class GE2ELoss(MetricLearningLoss):
    """
    Compute the sotfmax version of the GE2E (Generalized End To End) loss

    "Generalized End-to-End Loss for Speaker Verification",
    Wan et al., https://arxiv.org/abs/1710.10467
    """

    def __init__(self, embedding_size, n_classes, device="cpu"):
        super(GE2ELoss, self).__init__(embedding_size, n_classes, device=device)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def compute_similarity(self, embedding, centroid):
        """
        Compute the cosine similarity between embedding of speaker j,
        utterance i and the mean embedding (centroid) of speaker k
        """
        return F.relu(self.w) * F.cosine_similarity(embedding, centroid, dim=0) + self.b

    def compute_loss(self, speaker_embedding, speaker, embeddings_per_speaker):
        """
        Compute the softmax variant of GE2E loss for the given embedding
        e_ji, where j identifies the speaker and i the utterance
        """
        # Compute the first term of the summation, i.e. -s_ji,j
        centroid_minus_self = (
            embeddings_per_speaker[speaker].sum() - speaker_embedding
        ) / embeddings_per_speaker[speaker].shape[0]
        self_similarity = self.compute_similarity(
            speaker_embedding, centroid_minus_self
        )

        # Compute the second term of the summation, i.e. log(sum(exp(s_ji,k)))
        summation = torch.tensor(0.0, device=self.device, requires_grad=True)
        for other_speaker in embeddings_per_speaker:
            similarity = torch.clone(self_similarity)
            if other_speaker != speaker:
                centroid = (
                    embeddings_per_speaker[other_speaker].sum()
                    / embeddings_per_speaker[other_speaker].shape[0]
                )
                similarity = self.compute_similarity(speaker_embedding, centroid)
            summation = summation + torch.exp(similarity)

        # Sum first and second term
        return -self_similarity + torch.log(summation)

    def forward(self, inputs, targets):
        """
        Compute the softmax variant of GE2E loss for inputs
        of size [B, E] and targets of size [B] (it differs from
        the original implementation as it does not expect a fixed
        number of utterances per speaker as input)

        B: batch size
        E: embedding size
        """
        # Create dictionary of tensors u, s.t. u[k] is a matrix
        # containing all utterances of speaker k
        embeddings_per_speaker = dict()
        for speaker in torch.unique(targets):
            embeddings_per_speaker[speaker.item()] = torch.index_select(
                inputs, dim=0, index=(targets == speaker).nonzero(as_tuple=True)[0]
            )

        # Compute loss for each embedding e_ji
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for speaker in embeddings_per_speaker:
            for speaker_embedding in embeddings_per_speaker[speaker]:
                loss = loss + self.compute_loss(
                    speaker_embedding, speaker, embeddings_per_speaker
                )

        # Return total sum of losses
        return F.normalize(inputs, p=2, dim=1), None, loss


LOSSES = {
    "ce": CELoss,
    "sphere": SphereFaceLoss,
    "cos": CosFaceLoss,
    "arc": ArcFaceLoss,
    "ge2e": GE2ELoss,
}
