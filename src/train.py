import sys
from argparse import ArgumentParser
from functools import partial

import torch
import torch.optim as optim
import numpy as np
import yaml

import datasets, transforms, models, learn, losses, utils


def get_datasets(
    dataset_root,
    transformations=None,
    train_fraction=0.8,
    test_speakers=10,
    test_utterances_per_speaker=10,
):
    """
    Return an instance of the dataset specified in the given
    parameters, splitted into training, validation and test sets
    """
    # Get the dataset
    dataset = datasets.LibriSpeechDataset(dataset_root, transforms=transformations)

    # Get speakers
    utterances = dataset.speakers_utterances
    speakers = dataset.speakers

    # Get training utterances as the first percentage slice
    random_speakers = np.random.permutation(speakers)
    train_speakers = int(train_fraction * len(speakers))
    train_utterances = utils.flatten(
        [utterances[s] for s in random_speakers[:train_speakers]]
    )

    # Get test utterances by selecting the given number of
    # utterances for each of the given number of speakers
    remaining_speakers = random_speakers[train_speakers:]
    assert (
        len(remaining_speakers) > test_speakers
    ), "Not enough speakers for test and validation"
    test_utterances = utils.flatten(
        [
            utterances[s][:test_utterances_per_speaker]
            for s in remaining_speakers[:test_speakers]
        ]
    )

    # Get validation set utterances as the ones remaining after
    # training and test set splits
    val_utterances = []
    for s in remaining_speakers:
        min_utterance = test_utterances_per_speaker if s < test_speakers else 0
        val_utterances += utterances[s][min_utterance:]

    # Split dataset by speakers
    return (
        torch.utils.data.Subset(dataset, train_utterances),
        torch.utils.data.Subset(dataset, val_utterances),
        torch.utils.data.Subset(dataset, test_utterances),
        len(speakers),
    )


def get_random_dataloader(dataset, batch_size, num_workers=4, n_mels=80, seed=42):
    """
    Return a dataloader that randomly samples a batch of data
    from the given dataset, to use in the training procedure
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(datasets.collate_fn, n_mels=n_mels),
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=True,
    )


def get_sequential_dataloader(dataset, num_workers=4, n_mels=80, seed=42):
    """
    Return a dataloader that sequentially samples one observation
    at a time from the given dataset, to use in the validation phase
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(datasets.collate_fn, n_mels=n_mels),
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=True,
    )


def get_dataloaders(
    train_dataset, val_dataset, batch_size, num_workers=4, n_mels=80, seed=42
):
    """
    Return the appropriate dataloader for each dataset split
    """
    return (
        get_random_dataloader(
            train_dataset, batch_size, num_workers=num_workers, n_mels=n_mels, seed=seed
        ),
        get_sequential_dataloader(
            val_dataset, num_workers=num_workers, n_mels=n_mels, seed=seed
        ),
    )


def train(params):
    """
    Training loop entry-point
    """
    # Set random seed for reproducibility
    # and get GPU if present
    utils.set_seed(params.generic.seed)
    device = utils.get_device()

    # Set number of threads
    torch.set_num_threads(params.generic.workers)

    # Get data transformations
    transformations = transforms.get_transforms(
        params.augmentation.enable,
        params.augmentation.rir_corpora_path,
        max_length=params.augmentation.max_length,
        chunk_lengths=params.augmentation.chunk_lengths,
        min_speed=params.augmentation.min_speed,
        max_speed=params.augmentation.max_speed,
        sample_rate=params.audio.sample_rate,
        n_fft=params.audio.spectrogram.n_fft,
        win_length=params.audio.spectrogram.win_length,
        hop_length=params.audio.spectrogram.hop_length,
        n_mels=params.audio.spectrogram.n_mels,
    )

    # Get datasets and dataloaders
    train_dataset, val_dataset, test_dataset, n_speakers = get_datasets(
        params.dataset.root,
        transformations=transformations,
        train_fraction=params.training.train_fraction,
        test_speakers=params.test.num_speakers,
        test_utterances_per_speaker=params.test.num_utterances_per_speaker,
    )
    if params.training.dumb.enabled:
        train_dataset = test_dataset
    train_dataloader, val_dataloader = get_dataloaders(
        train_dataset,
        val_dataset,
        params.training.batch_size,
        num_workers=params.generic.workers,
        n_mels=params.audio.spectrogram.n_mels,
        seed=params.generic.seed,
    )

    # Get loss function
    loss_params = dict()
    if params.training.loss in params.loss.__dict__:
        loss_params = params.loss.__dict__[params.training.loss].__dict__["entries"]
    loss_function = losses.LOSSES[params.training.loss](
        params.titanet.embedding_size, n_speakers, **loss_params
    )

    # Get TitaNet model
    if params.training.dumb.enabled:
        model = models.DumbConvNet(
            params.audio.spectrogram.n_mels,
            loss_function,
            n_layers=params.training.dumb.n_layers,
        )
    else:
        n_mega_blocks = None
        if params.titanet.n_mega_blocks:
            n_mega_blocks = params.titanet.n_mega_blocks
        model = models.TitaNet.get_titanet(
            loss_function,
            embedding_size=params.titanet.embedding_size,
            n_mels=params.audio.spectrogram.n_mels,
            n_mega_blocks=n_mega_blocks,
            model_size=params.titanet.model_size,
            attention_hidden_size=params.titanet.attention_hidden_size,
            simple_pool=params.titanet.simple_pool,
            dropout=params.titanet.dropout,
            device=device,
        )

    # Use backprop to chart dependencies
    if params.generic.chart_dependencies:
        utils.chart_dependencies(
            model, n_mels=params.audio.spectrogram.n_mels, device=device
        )

    # Get optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=params.training.learning_rate)
    optimizer = utils.optimizer_to(optimizer, device=device)
    lr_scheduler = None
    if params.training.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_dataloader) * params.training.epochs
        )
        lr_scheduler = utils.scheduler_to(lr_scheduler, device=device)

    # Start wandb logging
    wandb_run = None
    run_name = utils.now()
    if params.wandb.enabled:
        wandb_run = utils.init_wandb(
            params.wandb.api_key_file,
            params.wandb.project,
            params.wandb.entity,
            name=run_name,
            config=params.entries,
        )

    # Perform training loop
    learn.training_loop(
        run_name,
        params.training.epochs,
        model,
        optimizer,
        train_dataloader,
        params.training.checkpoints_path,
        test_dataset=test_dataset if params.test.enabled else None,
        val_dataloader=val_dataloader,
        val_every=params.training.val_every or None,
        figures_path=params.training.figures_path,
        reduction_method=params.training.reduction_method,
        lr_scheduler=lr_scheduler,
        checkpoints_frequency=params.training.checkpoints_frequency,
        wandb_run=wandb_run,
        log_console=params.generic.log_console,
        mindcf_p_target=params.test.mindcf_p_target,
        mindcf_c_fa=params.test.mindcf_c_fa,
        mindcf_c_miss=params.test.mindcf_c_miss,
        device=device,
    )

    # Stop wandb logging
    if params.wandb.enabled:
        wandb_run.finish()


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
