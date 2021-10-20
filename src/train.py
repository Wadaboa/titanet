from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio
import numpy as np
import yaml

import datasets, transforms, model, learn, losses, utils


def get_transforms(
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
):
    """
    Return the list of transformations described in TitaNet paper
    """
    return [
        transforms.RandomChunk(max_length, chunk_lengths),
        transforms.SpeedPerturbation(min_speed, max_speed, probability=probability),
        transforms.NormalizedMelSpectrogram(
            sample_rate,
            n_fft=n_fft,
            win_length=int(win_length / 1000 * sample_rate),
            hop_length=int(hop_length / 1000 * sample_rate),
            n_mels=n_mels,
        ),
        transforms.SpecAugment(
            freq_mask_param=freq_mask_param,
            freq_mask_num=freq_mask_num,
            time_mask_param=time_mask_param,
            time_mask_num=time_mask_num,
            probability=probability,
        ),
    ]


def get_datasets(dataset_root, transforms, train_fraction=0.8, val_fraction=0.1):
    """
    Return an instance of the dataset specified in the given
    parameters, splitted into training, validation and test sets
    """
    assert train_fraction + val_fraction < 1.0, "Not enough data for test set"

    # Get the dataset
    dataset = datasets.LibriSpeechDataset(dataset_root, transforms=transforms)

    # Get speakers
    utterances = dataset.speakers_utterances
    speakers = dataset.speakers

    # Compute training and validation set sizes
    train_size = int(train_fraction * len(speakers))
    val_size = int(val_fraction * len(speakers))

    # Get training, validation and test indices
    random_speakers = np.random.permutation(speakers)
    train_indices = utils.flatten([utterances[i] for i in random_speakers[:train_size]])
    val_indices = utils.flatten(
        [utterances[i] for i in random_speakers[train_size : train_size + val_size]]
    )
    test_indices = utils.flatten(
        [utterances[i] for i in random_speakers[train_size + val_size :]]
    )

    # Split dataset by speakers
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices),
        len(speakers),
    )


def get_random_dataloader(dataset, batch_size, num_workers=1):
    """
    Return a dataloader that randomly samples a batch of data
    from the given dataset, to use in the training procedure
    """
    random_sampler = torch.utils.data.RandomSampler(dataset)
    random_batch_sampler = torch.utils.data.BatchSampler(
        random_sampler, batch_size, drop_last=False
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=random_batch_sampler,
        num_workers=num_workers,
        collate_fn=datasets.collate_fn,
    )


def get_sequential_dataloader(dataset, num_workers=1):
    """
    Return a dataloader that sequentially samples one observation
    at a time from the given dataset, to use in validation/test phases
    """
    sequential_sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sequential_sampler,
        num_workers=num_workers,
        collate_fn=datasets.collate_fn,
    )


def get_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size, num_workers=1
):
    """
    Return the appropriate dataloader for each dataset split
    """
    return (
        get_random_dataloader(train_dataset, batch_size, num_workers=num_workers),
        get_sequential_dataloader(val_dataset, num_workers=num_workers),
        get_sequential_dataloader(test_dataset, num_workers=num_workers),
    )


def train(params):
    """
    Training loop entry-point
    """
    # Get data transformations
    transforms = get_transforms(
        max_length=params.audio.augmentation.max_length,
        chunk_lengths=params.audio.augmentation.chunk_lengths,
        min_speed=params.audio.augmentation.min_speed,
        max_speed=params.audio.augmentation.max_speed,
        sample_rate=params.audio.sample_rate,
        n_fft=params.audio.spectrogram.n_fft,
        win_length=params.audio.spectrogram.win_length,
        hop_length=params.audio.spectrogram.hop_length,
        n_mels=params.audio.spectrogram.n_mels,
    )

    # Get datasets and dataloaders
    train_dataset, val_dataset, test_dataset, n_speakers = get_datasets(
        params.dataset.root,
        transforms,
        params.training.train_fraction,
        params.training.val_fraction,
    )
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        params.training.batch_size,
        params.generic.workers,
    )

    # Get loss function
    loss_params = dict()
    if params.training.loss in params.loss.__dict__:
        loss_params = params.loss.__dict__[params.training.loss].__dict__["entries"]
    loss_function = losses.LOSSES[params.training.loss](
        params.titanet.embedding_size, n_speakers, **loss_params
    )

    # Get TitaNet model
    device = utils.get_device()
    titanet = model.get_titanet(
        loss_function,
        embedding_size=params.titanet.embedding_size,
        n_mels=params.audio.spectrogram.n_mels,
        n_mega_blocks=params.titanet.n_mega_blocks,
        model_size=params.titanet.model_size,
        device=device,
    )

    # Get optimizer and scheduler
    optimizer = optim.SGD(titanet.parameters(), lr=params.training.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * params.training.epochs
    )

    # Perform training loop
    learn.training_loop(
        params.training.epochs,
        titanet,
        optimizer,
        train_dataloader,
        val_dataloader,
        params.training.checkpoints_path,
        lr_scheduler=lr_scheduler,
        checkpoints_frequency=params.training.checkpoints_frequency,
        wandb_enabled=params.wandb.enabled,
    )


if __name__ == "__main__":
    # Parse parameters
    parser = ArgumentParser(description="train the TitaNet model")
    parser.add_argument(
        "-p",
        "--params",
        help="path for the parameters .yml file",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    with open(args.params, "r") as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    params = utils.Struct(**args)

    # Call the training function
    train(params)
