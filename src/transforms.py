import random
import os
import shutil
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
import requests


def copy_example(example):
    """
    Deep copy the given dataset example
    """
    new_example = dict()
    for k, v in example.items():
        if isinstance(v, torch.Tensor):
            new_example[k] = torch.clone(v)
        else:
            new_example[k] = v
    return new_example


def get_transforms(
    enabled,
    rir_corpora_path,
    max_length=3,
    chunk_lengths=[1.5, 2, 3],
    min_speed=0.95,
    max_speed=1.05,
    sample_rate=16000,
    n_fft=512,
    win_length=25,
    hop_length=10,
    n_mels=80,
    freq_mask_param=100,
    freq_mask_num=1,
    time_mask_param=80,
    time_mask_num=1,
    probability=1.0,
    device="cpu",
):
    """
    Return the list of transformations described in TitaNet paper
    """
    transformations = [Resample(sample_rate)]
    if "chunk" in enabled:
        transformations += [RandomChunk(max_length, chunk_lengths)]
    if "reverb" in enabled:
        transformations += [
            Reverb(rir_corpora_path, probability=probability, device=device)
        ]
    transformations += [
        NormalizedMelSpectrogram(
            sample_rate,
            n_fft=n_fft,
            win_length=int(win_length / 1000 * sample_rate),
            hop_length=int(hop_length / 1000 * sample_rate),
            n_mels=n_mels,
        )
    ]
    if "specaugment" in enabled:
        transformations += [
            SpecAugment(
                min_speed,
                max_speed,
                freq_mask_param=freq_mask_param,
                freq_mask_num=freq_mask_num,
                time_mask_param=time_mask_param,
                time_mask_num=time_mask_num,
                probability=probability,
            )
        ]
    return transformations


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

        new_example = copy_example(example)
        if random.random() < self.probability:
            speed = random.uniform(self.min_speed, self.max_speed)
            (
                new_example["waveform"],
                new_example["sample_rate"],
            ) = torchaudio.sox_effects.apply_effects_tensor(
                new_example["waveform"],
                new_example["sample_rate"],
                [["speed", str(speed)], ["rate", str(new_example["sample_rate"])]],
            )

        return new_example


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

        new_example = copy_example(example)
        new_example["spectrogram"] = super().forward(new_example["waveform"])
        new_example["spectrogram"] = self.amplitude_to_db(new_example["spectrogram"])
        new_example["spectrogram"] = F.normalize(new_example["spectrogram"], dim=1)

        return new_example


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

        new_example = copy_example(example)
        num_samples = new_example["waveform"].size(-1)
        if num_samples / new_example["sample_rate"] > self.max_length:
            length = random.choice(self.lengths)
            samples = int(length * new_example["sample_rate"])
            start = random.randint(0, num_samples - samples)
            new_example["waveform"] = new_example["waveform"][
                :, start : start + samples
            ]

        return new_example


class SpecAugment:
    """
    Apply SpecAugment data augmentation, by randomly
    masking the frequency and/or time axes of the
    given mel-spectrogram
    """

    def __init__(
        self,
        min_time_stretch,
        max_time_stretch,
        freq_mask_param=100,
        freq_mask_num=1,
        time_mask_param=80,
        time_mask_num=1,
        probability=1.0,
    ):
        self.min_time_stretch = min_time_stretch
        self.max_time_stretch = max_time_stretch
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.probability = probability
        self.time_stretching = torchaudio.transforms.TimeStretch()
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

        new_example = copy_example(example)
        if random.random() < self.probability:
            time_stretch = random.uniform(self.min_time_stretch, self.max_time_stretch)
            new_example["spectrogram"] = self.time_stretching(
                new_example["spectrogram"], time_stretch
            )
            for _ in range(self.freq_mask_num):
                new_example["spectrogram"] = self.frequency_masking(
                    new_example["spectrogram"]
                )
            for _ in range(self.time_mask_num):
                new_example["spectrogram"] = self.time_masking(
                    new_example["spectrogram"]
                )

        return new_example


class Reverb:
    """
    Apply convolution reverb to the input signal to make a clean
    audio data sound like in a different environment
    """

    RIR_CORPORA_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"

    def __init__(self, rir_corpora_path, probability=1.0, device="cpu"):
        self.rir_corpora_path = rir_corpora_path
        self.probability = probability
        self.device = device
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

        new_example = copy_example(example)
        if random.random() < self.probability:
            # Extract random RIR and resample to input waveform
            # sampling rate
            sr = new_example["sample_rate"]
            rir_file = random.choice(self.rir_files)
            rir, rir_sr = torchaudio.load(rir_file)
            rir = rir.to(self.device)
            rir = torchaudio.functional.resample(rir, orig_freq=rir_sr, new_freq=sr)

            # Clean up the RIR: normalize the signal power and then flip the time axis
            rir = rir / torch.norm(rir, p=2)
            rir = torch.flip(rir, [1])

            # Convolve RIR with input waveform
            waveform = torch.nn.functional.pad(
                new_example["waveform"], (rir.shape[1] - 1, 0)
            )
            if waveform.size(0) == rir.size(0):
                new_example["waveform"] = torch.nn.functional.conv1d(
                    waveform[None, :, :], rir[None, :, :]
                )[0]

        return new_example


class ToDevice:
    """
    Transfer tensors to device
    """

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, example):
        assert isinstance(example, dict), "Wrong input structure"

        new_example = copy_example(example)
        for k, v in new_example.items():
            new_example[k] = v
            if isinstance(v, torch.Tensor):
                new_example[k] = v.to(self.device)
        return new_example


class Resample:
    """
    Change the sample rate of the raw waveform to the target one
    """

    def __init__(self, target_sample_rate):
        self.target_sample_rate = target_sample_rate

    def __call__(self, example):
        assert (
            isinstance(example, dict)
            and "waveform" in example
            and "sample_rate" in example
        ), "Wrong input structure"

        new_example = copy_example(example)
        new_example["waveform"] = torchaudio.functional.resample(
            new_example["waveform"], new_example["sample_rate"], self.target_sample_rate
        )
        new_example["sample_rate"] = self.target_sample_rate

        return new_example
