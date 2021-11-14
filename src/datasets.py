import os
import itertools
from collections import defaultdict
from functools import partial

import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

import utils


def get_dataloader(
    dataset, batch_size=1, shuffle=True, num_workers=4, n_mels=80, seed=42
):
    """
    Return a dataloader that randomly (or sequentially) samples a batch
    of data from the given dataset
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, n_mels=n_mels),
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=True,
        drop_last=True,
    )


def collate_fn(batch, n_mels=80):
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
    spectrogram_lengths = torch.LongTensor(spectrogram_lengths)
    speakers = torch.LongTensor(speakers)

    # Fill tensors up to the maximum length
    spectrograms = torch.zeros(
        len(batch), n_mels, max(spectrogram_lengths), dtype=torch.float32
    )
    for i, example in enumerate(batch):
        spectrograms[i, :, : spectrogram_lengths[i]] = example["spectrogram"].to(
            torch.float32
        )

    # Return a batch of tensors
    return spectrograms, spectrogram_lengths, speakers


def get_datasets(
    dataset_root,
    train_transformations=None,
    non_train_transformations=None,
    val=True,
    val_utterances_per_speaker=10,
    test=True,
    test_speakers=10,
    test_utterances_per_speaker=10,
):
    """
    Return an instance of the dataset specified in the given
    parameters, splitted into training, validation and test sets
    """
    # Get the dataset
    full_dataset = LibriSpeechDataset(dataset_root)

    # Compute train, validation and test utterances
    train_utterances, val_utterances, test_utterances = full_dataset.get_splits(
        val=val,
        val_utterances_per_speaker=val_utterances_per_speaker,
        test=test,
        test_speakers=test_speakers,
        test_utterances_per_speaker=test_utterances_per_speaker,
    )

    # Split dataset
    train_dataset = full_dataset.subset(
        train_utterances, transforms=train_transformations
    )
    val_dataset = full_dataset.subset(
        val_utterances, transforms=non_train_transformations
    )
    test_dataset = full_dataset.subset(
        test_utterances, transforms=non_train_transformations
    )

    return train_dataset, val_dataset, test_dataset, full_dataset.get_num_speakers()


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

    def get_path(self, idx):
        """
        Return the system path identifying the utterance at the given index
        """
        raise NotImplementedError()

    def get_random_utterances(self, n_speakers=5, n_utterances_per_speaker=20):
        """
        Return a list of random utterance ids for each random speaker
        """
        utterances, speakers = [], []
        random_speakers = np.random.choice(self.speakers, size=n_speakers)
        for speaker in random_speakers:
            speaker_utterances = self.speakers_utterances[speaker]
            utterances += list(
                np.random.choice(speaker_utterances, size=n_utterances_per_speaker)
            )
            speakers += [speaker] * n_utterances_per_speaker
        return utterances, speakers

    def get_sample_pairs(self, indices=None, device="cpu"):
        """
        Return a list of tuples (s1, s2, l), where s1, s2
        is a pair of utterances and l indicates whether
        w1 and w2 are from the same speaker
        """
        indices = indices or list(range(len(self)))
        indices = itertools.product(indices, repeat=2)
        samples = []
        for i1, i2 in tqdm(indices, desc="Loading sample pairs"):
            e1, e2 = self.__getitem__(i1), self.__getitem__(i2)
            samples += [
                (
                    e1["spectrogram"].to(device),
                    e2["spectrogram"].to(device),
                    e1["speaker"] == e2["speaker"],
                )
            ]
        return samples

    def get_num_speakers(self):
        """
        Return the number of speakers in the dataset
        """
        return len(self.speakers)

    def get_splits(
        self,
        val=True,
        val_utterances_per_speaker=10,
        test=True,
        test_speakers=10,
        test_utterances_per_speaker=10,
    ):
        """
        Return train, validation and test indices
        """
        # Split dataset
        train_utterances, val_utterances, test_utterances = [], [], []
        for i, s in enumerate(self.speakers):
            train_start_utterance = 0
            if val:
                val_utterances += self.speakers_utterances[s][
                    :val_utterances_per_speaker
                ]
                train_start_utterance += val_utterances_per_speaker
            if test and i < test_speakers:
                test_utterances += self.speakers_utterances[s][
                    val_utterances_per_speaker : val_utterances_per_speaker
                    + test_utterances_per_speaker
                ]
                train_start_utterance += test_utterances_per_speaker
            train_utterances += self.speakers_utterances[s][train_start_utterance:]

        # Check split correctness
        assert (not val or len(val_utterances) > 0) and (
            not test or len(test_utterances) > 0
        ), "No validation or test utterances"
        assert not utils.overlap(
            train_utterances, val_utterances
        ) and not utils.overlap(
            val_utterances, test_utterances
        ), "Splits are not disjoint"

        return train_utterances, val_utterances, test_utterances

    def subset(self, indices, transforms=None):
        """
        Return a subset of the current dataset and possibly
        overwrite transformations
        """
        dataset = torch.utils.data.Subset(self, indices)
        dataset.dataset.transforms = transforms
        return dataset

    def get_durations(self):
        """
        Return the duration (in seconds) for each utterance
        """
        durations = dict()
        for idx in tqdm(range(len(self)), desc="Computing durations"):
            filename = self.get_path(idx)
            durations[idx] = librosa.get_duration(filename=filename)
        return durations

    def get_durations_per_speaker(self, hours=True):
        """
        Return a dictionary having as key a speaker id
        and as value the total number of seconds (or hours)
        for that speaker
        """
        durations = self.get_durations()
        durations_per_speaker = dict()
        div = 1 if not hours else 3600
        for speaker, utterances in self.speakers_utterances.items():
            durations_per_speaker[speaker] = (
                sum(durations[idx] for idx in utterances) / div
            )
        return durations_per_speaker

    def info(self, hours=True):
        """
        Return a dictionary of generic info about the dataset
        """
        utterances_per_speaker = [len(u) for u in self.speakers_utterances.values()]
        durations_per_speaker = list(
            self.get_durations_per_speaker(hours=hours).values()
        )
        return {
            "num_utterances": len(self),
            "num_speakers": self.get_num_speakers(),
            "total_duration": round(sum(durations_per_speaker), 2),
            "utterances_per_speaker_mean": round(np.mean(utterances_per_speaker), 2),
            "utterances_per_speaker_std": round(np.std(utterances_per_speaker), 2),
            "durations_per_speaker_mean": round(np.mean(durations_per_speaker), 2),
            "durations_per_speaker_std": round(np.std(durations_per_speaker), 2),
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
            speakers_utterances[int(speaker_id)].append(i)
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

    def get_path(self, idx):
        fileid = self._walker[idx]
        speaker_id, chapter_id, utterance_id = fileid.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + self._ext_audio
        return os.path.join(self._path, speaker_id, chapter_id, file_audio)


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

    def get_path(self, idx):
        speaker_id, utterance_id = self._sample_ids[idx]
        return os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{self._mic_id}{self._audio_ext}",
        )
