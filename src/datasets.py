import os
from collections import defaultdict

import torch
import torchaudio
import numpy as np


def collate_fn(batch, n_mels=80, device="cpu"):
    """
    Convert a list of samples extracted from the dataset to
    a proper batch, i.e. a tuple of stacked tensors
    """
    # Compute lengths and convert data to lists
    spectrogram_lengths, speakers = [], []
    for example in batch:
        spectrogram_lengths.append(example["spectrogram"].size(-1))
        speakers.append(example["speaker_id"])

    # Convert collected lists to tensors
    spectrogram_lengths = torch.LongTensor(spectrogram_lengths).to(device)
    speakers = torch.LongTensor(speakers).to(device)

    # Fill tensors up to the maximum length
    spectrograms = torch.zeros(
        len(batch), n_mels, max(spectrogram_lengths), dtype=torch.float, device=device
    )
    for i, example in enumerate(batch):
        spectrograms[i, :, : spectrogram_lengths[i]] = (
            example["spectrogram"].to(device).to(torch.float32)
        )

    # Return a batch of tensors
    return spectrograms, spectrogram_lengths, speakers


class SpeakerDataset:
    """
    Generic dataset with the ability to apply transforms
    on raw input data, that returns a standard output
    when indexed
    """

    def __init__(self, transforms=None):
        self.transforms = transforms or []
        self.speakers_utterances = self.get_speakers_utterances()
        self.speakers = list(self.speakers_utterances.keys())
        self.speakers_to_id = dict(zip(self.speakers, range(len(self.speakers))))
        self.id_to_speakers = dict(zip(range(len(self.speakers)), self.speakers))

    def get_speakers_utterances(self):
        """
        Return a dictionary having as key a speaker id and
        as value a list of indices that identify the
        utterances spoken by that speaker
        """
        raise NotImplementedError()

    def get_sample(self, idx):
        """
        Return a tuple (waveform, rate, speaker) identifying
        the utterance at the given index
        """
        raise NotImplementedError()

    def info(self):
        """
        Return a dictionary of generic info about the dataset
        """
        utterances_per_speaker = [len(u) for u in self.speakers_utterances.values()]
        return {
            "utterances": len(self),
            "speakers": len(self.speakers),
            "utterances_per_speaker_mean": round(np.mean(utterances_per_speaker), 2),
            "utterances_per_speaker_std": round(np.std(utterances_per_speaker), 2),
        }

    def __getitem__(self, idx):
        waveform, sample_rate, speaker = self.get_sample(idx)
        example = {
            "waveform": waveform.to("cpu"),
            "sample_rate": sample_rate,
            "spectrogram": None,
            "speaker": speaker,
            "speaker_id": self.speakers_to_id[speaker],
        }
        for transform in self.transforms:
            example = transform(example)
        return example


class LibriSpeechDataset(SpeakerDataset, torchaudio.datasets.LIBRISPEECH):
    """
    Custom LibriSpeech dataset for speaker-related tasks
    """

    def __init__(self, root, transforms=None, *args, **kwargs):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            kwargs["download"] = True
        torchaudio.datasets.LIBRISPEECH.__init__(self, root, *args, **kwargs)
        SpeakerDataset.__init__(self, transforms=transforms)

    def get_speakers_utterances(self):
        speakers_utterances = defaultdict(list)
        for i, fileid in enumerate(self._walker):
            speaker_id, _, _ = fileid.split("-")
            speakers_utterances[speaker_id].append(i)
        return speakers_utterances

    def get_sample(self, idx):
        (
            waveform,
            sample_rate,
            _,
            speaker,
            _,
            _,
        ) = torchaudio.datasets.LIBRISPEECH.__getitem__(self, idx)
        return waveform, sample_rate, speaker


class VCTKDataset(SpeakerDataset, torchaudio.datasets.VCTK_092):
    """
    Custom VCTK dataset for speaker-related tasks
    """

    def __init__(self, root, transforms=None, *args, **kwargs):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            kwargs["download"] = True
        torchaudio.datasets.VCTK_092.__init__(self, root, *args, **kwargs)
        SpeakerDataset.__init__(self, transforms=transforms)

    def get_speakers_utterances(self):
        speakers_utterances = defaultdict(list)
        for i, (speaker_id, _) in enumerate(self._sample_ids):
            speakers_utterances[speaker_id].append(i)
        return speakers_utterances

    def get_sample(self, idx):
        waveform, sample_rate, _, speaker, _ = torchaudio.datasets.VCTK_092.__getitem__(
            self, idx
        )
        return waveform, sample_rate, speaker
