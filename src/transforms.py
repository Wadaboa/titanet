import random

import torch
import torchaudio
import torch.nn.functional as F


class SpeedPerturbation:
    """
    Randomly change the speed of the given waveform,
    with the speed ranges specified in input
    """

    def __init__(
        self,
        min_speed,
        max_speed,
        probability=1.0,
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.probability = probability

    def __call__(self, example):
        assert (
            isinstance(example, dict)
            and "waveform" in example
            and "sample_rate" in example
        ), "Wrong input structure"

        if random.random() < self.probability:
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


class SpecAugment:
    """
    Apply SpecAugment data augmentation, by randomly
    masking the frequency and/or time axes of the
    given mel-spectrogram (no time stretch)
    """

    def __init__(
        self,
        freq_mask_param=100,
        freq_mask_num=1,
        time_mask_param=80,
        time_mask_num=1,
        probability=1.0,
    ):
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.probability = probability
        self.frequency_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param, iid_masks=True
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param, iid_masks=True
        )

    def __call__(self, example):
        assert (
            isinstance(example, dict) and "spectrogram" in example
        ), "Wrong input structure"

        if random.random() < self.probability:
            for _ in range(self.freq_mask_num):
                example["spectrogram"] = self.frequency_masking(example["spectrogram"])
            for _ in range(self.time_mask_num):
                example["spectrogram"] = self.time_masking(example["spectrogram"])
        return example
