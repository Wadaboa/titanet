import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio

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
):
    return [
        transforms.RandomChunk(max_length, chunk_lengths),
        transforms.SpeedPerturbation(min_speed, max_speed),
        transforms.NormalizedMelSpectrogram(
            sample_rate,
            n_fft=n_fft,
            win_length=win_length / 1000 * sample_rate,
            hop_length=hop_length / 1000 * sample_rate,
            n_mels=n_mels,
        ),
    ]


def get_datasets(dataset_root, transforms, train_fraction=0.8, val_fraction=0.1):
    """
    Return an instance of the dataset specified in the given
    parameters, splitted into training, validation and test sets
    """
    dataset = datasets.LibriSpeechDataset(dataset_root, transforms=transforms)
    train_size = int(train_fraction * len(dataset))
    val_size = int(val_fraction * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset


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
        collate_fn=datasets.LibriSpeechDataset.collate_fn,
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
        collate_fn=datasets.LibriSpeechDataset.collate_fn,
    )


def get_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size, num_workers=1
):
    """
    Return the appropriate dataloader for each dataset split
    """
    return (
        get_random_dataloader(train_dataset, batch_size, num_workers),
        get_sequential_dataloader(val_dataset, num_workers),
        get_sequential_dataloader(test_dataset, num_workers),
    )


def train():
    transforms = get_transforms()
    train_dataset, val_dataset, test_dataset = get_datasets(
        transforms, train_fraction, val_fraction
    )
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )

    device = utils.get_device()
    loss_function = losses.LOSSES[loss_type]()
    titanet = model.get_titanet(
        loss_function,
        n_mels=n_mels,
        n_mega_blocks=n_mega_blocks,
        model_size=model_size,
        device=device,
    )

    optimizer = optim.SGD(titanet.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * training_epochs
    )

    learn.training_loop(
        epochs,
        titanet,
        optimizer,
        train_dataloader,
        val_dataloader,
        checkpoints_path,
        lr_scheduler=lr_scheduler,
        checkpoints_frequency=checkpoints_frequency,
        wandb_enabled=wandb_enabled,
    )


if __name__ == "__main__":
    train()
