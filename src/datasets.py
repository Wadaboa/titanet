import os
import itertools
import csv
import shutil
import logging
from pathlib import Path
from collections import defaultdict
from functools import partial

import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from librosa.core.audio import __audioread_load as audioread_load
from tqdm import tqdm
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive

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


class VoxCeleb1Dataset(SpeakerDataset, torchaudio.datasets.VoxCeleb1Identification):
    """
    Custom VoxCeleb1 dataset for speaker-related tasks
    """

    def __init__(self, root, transforms=None, *args, **kwargs):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            kwargs["download"] = True
        torchaudio.datasets.VoxCeleb1Identification.__init__(
            self, root, *args, **kwargs
        )
        SpeakerDataset.__init__(self, transforms=transforms)

    def get_speakers_utterances(self):
        speakers_utterances = defaultdict(list)
        for i, file_path in enumerate(self._flist):
            speaker_id, _, _ = file_path.split("/")[-3:]
            speakers_utterances[speaker_id].append(i)
        return speakers_utterances

    def get_sample(self, idx):
        (
            waveform,
            sample_rate,
            speaker,
            _,
        ) = torchaudio.datasets.VoxCeleb1Identification.__getitem__(self, idx)
        return waveform, sample_rate, speaker

    def get_path(self, idx):
        return self._flist[idx]


class VoxCeleb2(Dataset):
    """
    VoxCeleb2 dataset following torchaudio's implementation of VoxCeleb1.

    References:
    - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
    - https://pytorch.org/audio/stable/_modules/torchaudio/datasets/voxceleb1.html
    """

    SAMPLE_RATE = 16000
    # Credentials from https://github.com/UoA-CARES-Student/VoxCeleb2-Dataset
    _USERNAME = "voxceleb1912"
    _PASSWORD = "0s42xuw6"
    _ARCHIVE_CONFIGS = {
        "dev": {
            "archive_name": "vox2_dev_aac.zip",
            "urls": [
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partaa",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partab",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partac",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partad",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partae",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partaf",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partag",
                "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partah",
            ],
            "checksums": [None, None, None, None, None, None, None, None],
        },
        "test": {
            "archive_name": "vox2_test_aac.zip",
            "url": "http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_aac.zip",
            "checksum": "e4d9200107a7bc60f0b620d5dc04c3aab66681b649f9c218380ac43c6c722079",
        },
    }
    _IDEN_SPLIT_URL = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv"
    _ext_audio = ".wav"

    def __init__(self, root, subset="dev", meta_url=_IDEN_SPLIT_URL, download=False):
        if subset not in ["dev", "test"]:
            raise ValueError("`subset` must be one of ['dev', 'test']")
        root = os.fspath(root)
        self._path = os.path.join(root, "wav")
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. Please set `download=True` to download the dataset."
                )
            self._download_extract_wavs(root)

        # Download the `vox2_meta.csv` file to get the dev and test lists
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = self._get_flist(root, meta_list_path, subset)

    def _convert_to_wav(self, root, paths):
        """
        Convert .m4a files in the given paths to .wav
        """

        def _to_wav(path):
            try:
                waveform, _ = audioread_load(
                    path, offset=0.0, duration=None, dtype=np.float32
                )
                path_wav = os.path.splitext(path)[0] + ".wav"
                sf.write(path_wav, waveform, self.SAMPLE_RATE)
            except:
                logging.warning(f"Could not convert file {path} to .wav.")
            os.remove(path)

        Parallel(n_jobs=-1, backend="threading")(
            delayed(_to_wav)(os.path.join(root, p))
            for p in tqdm(paths, desc="Converting audios to .wav")
            if p.endswith(".m4a")
        )

    def _download_extract_wavs(self, root):
        """
        Download dataset splits, extract zipped archives
        and convert .m4a files to .wav
        """
        if not os.path.isdir(root):
            os.makedirs(root)
        for split, split_config in self._ARCHIVE_CONFIGS.items():
            split_name = split_config["archive_name"]
            split_path = os.path.join(root, split_name)
            # The zip file of dev data is splited to 8 chunks.
            # Download and combine them into one file before extraction.
            if split == "dev":
                urls = split_config["urls"]
                checksums = split_config["checksums"]
                with open(split_path, "wb") as f:
                    for url, checksum in zip(urls, checksums):
                        file_path = os.path.join(root, os.path.basename(url))
                        utils.download_auth_url_to_file(
                            url,
                            file_path,
                            self._USERNAME,
                            self._PASSWORD,
                            hash_prefix=checksum,
                        )
                        with open(file_path, "rb") as f_split:
                            f.write(f_split.read())
            elif split == "test":
                url = split_config["url"]
                checksum = split_config["checksum"]
                file_path = os.path.join(root, os.path.basename(url))
                utils.download_auth_url_to_file(
                    url, file_path, self._USERNAME, self._PASSWORD, hash_prefix=checksum
                )
            extracted_paths = extract_archive(split_path)
            self._convert_to_wav(root, extracted_paths)
        shutil.move(os.path.join(root, "aac"), os.path.join(root, "wav"))

    def _get_flist(self, root, meta_list_path, subset):
        """
        Load the full list of files in the given split
        """
        f_list = []
        with open(meta_list_path, "r") as f:
            csv_file = csv.reader(f, delimiter=",")
            for line in csv_file:
                id, set = line[0].strip(), line[-1].strip()
                if set == subset:
                    f_list += [str(i) for i in Path(root).rglob(f"{id}/**/*.wav")]
        return sorted(f_list)

    def _get_file_id(self, file_path, _ext_audio):
        """
        Return the file identifier as a combination of speaker id,
        youtube video id and utterance id
        """
        speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
        utterance_id = utterance_id.replace(_ext_audio, "")
        file_id = "-".join([speaker_id, youtube_id, utterance_id])
        return file_id

    def get_metadata(self, n):
        """
        Get metadata for the n-th sample from the dataset.
        Returns filepath instead of waveform, but otherwise
        returns the same fields as `__getitem__`.
        """
        file_path = self._flist[n]
        file_id = self._get_file_id(file_path, self._ext_audio)
        speaker_id = file_id.split("-")[0]
        speaker_id = int(speaker_id[3:])
        return file_path, self.SAMPLE_RATE, speaker_id, file_id

    def __getitem__(self, n):
        """
        Load the n-th sample from the dataset
        """
        metadata = self.get_metadata(n)
        waveform, sample_rate = torchaudio.load(metadata[0], metadata[1])
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(
                f"sample rate should be {self.SAMPLE_RATE}, but got {sample_rate}"
            )
        return (waveform,) + metadata[1:]

    def __len__(self):
        return len(self._flist)


class VoxCeleb2Dataset(SpeakerDataset, VoxCeleb2):
    """
    Custom VoxCeleb2 dataset for speaker-related tasks
    """

    def __init__(self, root, transforms=None, *args, **kwargs):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            kwargs["download"] = True
        VoxCeleb2.__init__(self, root, *args, **kwargs)
        SpeakerDataset.__init__(self, transforms=transforms)

    def get_speakers_utterances(self):
        speakers_utterances = defaultdict(list)
        for i, file_path in enumerate(self._flist):
            speaker_id, _, _ = file_path.split("/")[-3:]
            speakers_utterances[speaker_id].append(i)
        return speakers_utterances

    def get_sample(self, idx):
        (
            waveform,
            sample_rate,
            speaker,
            _,
        ) = VoxCeleb2.__getitem__(self, idx)
        return waveform, sample_rate, speaker

    def get_path(self, idx):
        return self._flist[idx]
