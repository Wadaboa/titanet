import random
import os
import shutil
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
import requests


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


class Reverb:
    """
    Apply convolution reverb to the input signal to make a clean
    audio data sound like in a different environment
    """

    RIR_CORPORA_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"

    def __init__(self, rir_corpora_path, probability=1.0):
        self.rir_corpora_path = rir_corpora_path
        self.probability = probability
        if not os.path.exists(rir_corpora_path):
            self._download_rir_corpora()
        self.rir_files = list(Path(self.rir_corpora_path).rglob("*.wav"))
        assert (
            len(self.rir_files) > 0
        ), "There are no .wav files in the selected RIR corpora"

    def _download_rir_corpora(self):
        """
        Download the "Room Impulse Response and Noise Database"
        corpora, as described in TitaNet paper
        """
        os.makedirs(self.rir_corpora_path, exist_ok=True)
        print("Downloading RIR corpora...")
        content = requests.get(self.RIR_CORPORA_URL).content
        print("RIR corpora downloaded")
        file_path = os.path.join(self.rir_corpora_path, "rirs_noises.zip")
        with open(file_path, "wb") as file:
            file.write(content)
        shutil.unpack_archive(file_path, self.rir_corpora_path)
        os.remove(file_path)

    def __call__(self, example):
        assert (
            isinstance(example, dict) and "spectrogram" in example
        ), "Wrong input structure"

        if random.random() < self.probability:
            # Extract random RIR and resample to input waveform
            # sampling rate
            sr = example["sample_rate"]
            rir_file = random.choice(self.rir_files)
            rir, rir_sr = torchaudio.load(rir_file)
            rir = torchaudio.functional.resample(rir, orig_freq=rir_sr, new_freq=sr)

            # Clean up the RIR: normalize the signal power and then flip the time axis
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])

            # Convolve RIR with input waveform
            waveform = torch.nn.functional.pad(
                example["waveform"], (rir.shape[1] - 1, 0)
            )
            example["waveform"] = torch.nn.functional.conv1d(
                waveform[None, :, :], rir[None, :, :]
            )[0]

        return example
