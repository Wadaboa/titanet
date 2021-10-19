import torch
import torch.nn as nn
import torch.nn.functional as F

import modules


class Encoder(nn.Module):
    def __init__(
        self,
        n_mels,
        n_mega_blocks,
        n_sub_blocks,
        output_size,
        mega_block_kernel_size,
        prolog_kernel_size=3,
        epilog_kernel_size=1,
        dropout=0.5,
    ):
        super(Encoder, self).__init__()
        self.prolog = modules.ConvBlock1d(n_mels, output_size, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(
                    output_size,
                    output_size,
                    mega_block_kernel_size,
                    n_sub_blocks,
                    dropout=dropout,
                )
                for _ in range(n_mega_blocks)
            ]
        )
        self.epilog = modules.ConvBlock1d(output_size, output_size, epilog_kernel_size)

    def forward(self, x):
        x = self.prolog(x)
        x = self.mega_blocks(x)
        return self.epilog(x)


class MegaBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        n_sub_blocks,
        dropout=0.5,
        se_reduction=16,
    ):
        super(MegaBlock, self).__init__()

        self.dropout = dropout

        channels = [input_size] + [output_size] * n_sub_blocks
        self.sub_blocks = nn.Sequential(
            *[
                modules.ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation="relu",
                    dropout=dropout,
                    depthwise=True,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            modules.SqueezeExcitation(output_size, reduction=se_reduction)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, x):
        y = self.skip_connection(x) + self.sub_blocks(x)
        return F.dropout(F.relu(y), p=self.dropout, training=self.training)


class Decoder(nn.Module):
    def __init__(self, encoder_size, attention_hidden_size, embedding_size):
        super(Decoder, self).__init__()
        self.attention = nn.Sequential(
            AttentiveStatsPooling(encoder_size, attention_hidden_size),
            nn.BatchNorm1d(encoder_size * 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(encoder_size * 2, embedding_size), nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.attention(x)
        return self.linear(x)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_size, hidden_size, activation="relu"):
        super(AttentiveStatsPooling, self).__init__()

        # Store attributes
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define architecture
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.scores_w = nn.Parameter(torch.randn(hidden_size))
        self.scores_b = nn.Parameter(torch.zeros(1))

    def forward(self, spectrogram):
        # Transpose input spectrogram
        # [B, DE, T] -> [B, T, DE]
        spectrogram = spectrogram.transpose(1, 2)
        batch_size, _, _ = spectrogram.shape

        # Compute attention scores of shape [B, T]
        projection = self.activation(self.linear(spectrogram))
        scores_w = (
            self.scores_w.unsqueeze(0).expand(batch_size, self.hidden_size).unsqueeze(2)
        )
        scores = projection.bmm(scores_w).squeeze() + self.scores_b

        # Compute attention weights of shape [B, T]
        weights = F.softmax(scores, dim=1)

        # Compute attention context of shape [B, T, DE]
        context = torch.mul(
            spectrogram, weights.unsqueeze(2).expand(-1, -1, self.input_size)
        )

        # Compute pooling statistics (mean and variance)
        # each one of shape [B, DE]
        mean = torch.mean(context, dim=1)
        std = torch.sum(torch.mul(spectrogram, context), dim=1) - torch.mul(mean, mean)

        # Return the concatenation of pooling statistics
        # of shape [B, DE * 2]
        return torch.cat((mean, std), dim=1)
