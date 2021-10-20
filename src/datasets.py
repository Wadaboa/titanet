import os
from collections import defaultdict

import torch
import torchaudio


class LibriSpeechDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, root, transforms=None, *args, **kwargs):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        super(LibriSpeechDataset, self).__init__(root, *args, **kwargs)
        self.transforms = transforms or []

    def get_speakers_utterances(self):
        """
        Return a dictionary having as key a speaker id and
        as value a list of indices that identify the
        utterances spoken by that speaker
        """
        speakers_utterances = defaultdict(list)
        for i, fileid in enumerate(self._walker):
            speaker_id, _, _ = fileid.split("-")
            speakers_utterances[int(speaker_id)].append(i)
        return speakers_utterances

    @classmethod
    def collate_fn(cls, batch):
        """
        Convert a list of samples extracted from the dataset to
        a proper batch, i.e. a tuple of stacked tensors
        """
        # Compute lengths and convert data to lists
        spectrograms, spectrogram_lengths, speakers = [], []
        for example in batch:
            spectrograms.append(example["spectrogram"])
            spectrogram_lengths.append(len(example["spectrogram"]))
            speakers.append(example["speaker"])

        # Convert collected lists to tensors
        spectrogram_lengths = torch.LongTensor(spectrogram_lengths)
        speakers = torch.LongTensor(speakers)

        # Fill tensors up to the maximum length
        n_mels = spectrograms[0].size(1)
        spectrograms = torch.zeros(
            len(batch), n_mels, max(spectrogram_lengths), dtype=torch.float
        )
        for i, example in enumerate(batch):
            spectrogram = example["spectrogram"]
            spectrograms[i, :, : spectrogram.shape[-1]] = torch.FloatTensor(spectrogram)

        # Return a batch of tensors
        return spectrograms, spectrogram_lengths, speakers

    def __getitem__(self, idx):
        waveform, sample_rate, _, speaker, _, _ = super().__getitem__(idx)
        example = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "spectrogram": None,
            "speaker": speaker,
        }
        for transform in self.transforms:
            example = transform(example)
        return example
