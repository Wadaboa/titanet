import random

import torch
import torchaudio
import torch.nn.functional as F


class SpeedPerturbation:
    """
    Randomly change the speed of the given waveform,
    with the speed ranges specified in input
    """

    def __init__(self, min_speed, max_speed):
        self.min_speed = min_speed
        self.max_speed = max_speed

    def __call__(self, example):
        assert (
            isinstance(example, dict)
            and "waveform" in example
            and "sample_rate" in example
        ), "Wrong input structure"
        speed = random.uniform(self.min_speed, self.max_speed)
        (
            example["waveform"],
            example["sample_rate"],
        ) = torchaudio.sox_effects.apply_effects_tensor(
            example["waveform"], example["sample_rate"], [["speed", str(speed)]]
        )
        return example


class NormalizedMelSpectrogram(torchaudio.transforms.MelSpectrogram):
    """
    Compute mel-spectrograms from input waveforms, then transform
    amplitudes to decibels and normalize over the frequency dimension
    """

    def __init__(self, *args, **kwargs):
        super(NormalizedMelSpectrogram, self).__init__(*args, **kwargs)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, example):
        assert (
            isinstance(example, dict) and "waveform" in example
        ), "Wrong input structure"
        example["spectrogram"] = super().forward(example["waveform"])
        example["spectrogram"] = self.amplitude_to_db(example["spectrogram"])
        example["spectrogram"] = F.normalize(example["spectrogram"], dim=1)
        return example


class RandomChunk:
    """
    Extract random speech chunks of the given lengths, if
    the input waveform exceeds the given maximum length
    """

    def __init__(self, max_length, lengths):
        self.max_length = max_length
        self.lengths = lengths

    def __call__(self, example):
        assert (
            isinstance(example, dict)
            and "waveform" in example
            and "sample_rate" in example
        ), "Wrong input structure"
        num_samples = example["waveform"].size(-1)
        if num_samples / example["sample_rate"] > self.max_length:
            length = random.choice(self.lengths)
            samples = int(length * example["sample_rate"])
            start = random.randint(0, num_samples - samples)
            example["waveform"] = example["waveform"][:, start : start + samples]
        return example
