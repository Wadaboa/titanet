import os
import time
import datetime
import math
import sys

import torch
import wandb

import utils


def train_one_epoch(model, optimizer, dataloader):
    """
    Train the given model for one epoch with the given dataloader and optimizer
    """
    # Put the model in training mode
    model.train()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets = [], []
    for spectrograms, _, speakers in dataloader:

        # Forward pass
        model_time = time.time()
        preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_time += model_time
        epoch_preds += preds.detach().cpu().tolist()
        epoch_targets += speakers.detach().cpu().tolist()

        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get metrics and return them
    metrics = utils.get_metrics(epoch_targets, epoch_preds, prefix="train")
    metrics["loss"] = epoch_loss
    metrics["time"] = epoch_time
    return metrics


def save_checkpoint(
    epoch, checkpoints_path, model, optimizer, lr_scheduler=None, wandb_enabled=False
):
    """
    Save the current state of the model, optimizer and learning rate scheduler,
    both locally and on wandb (if available and enabled)
    """
    # Create state dictionary
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else dict()
        ),
        "epoch": epoch,
    }

    # Save state dictionary
    checkpoint_file = os.path.join(checkpoints_path, f"{utils.now()}_{epoch}.pth")
    torch.save(state_dict, checkpoint_file)
    if wandb_enabled:
        wandb.save(checkpoint_file)


def training_loop(
    epochs,
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    checkpoints_path,
    val_every,
    lr_scheduler=None,
    checkpoints_frequency=None,
    wandb_run=None,
):
    """
    Standard training loop function: train and evaluate
    after each training epoch
    """
    # Track execution time
    start_time = time.time()

    # For each epoch
    for epoch in range(1, epochs + 1):

        # Train for one epoch and log metrics to wandb
        train_metrics = train_one_epoch(model, optimizer, train_dataloader)
        if wandb_run is not None:
            wandb_run.log(train_metrics, step=epoch)

        # Decay the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoints once in a while
        if checkpoints_frequency is not None and epoch % checkpoints_frequency == 0:
            save_checkpoint(
                epoch, checkpoints_path, model, optimizer, lr_scheduler=lr_scheduler
            )

        # Evaluate once in a while (always evaluate at the first and last epochs)
        if epoch % val_every == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate(model, val_dataloader)
            if wandb_run is not None:
                wandb_run.log(val_metrics, step=epoch)

    # Always save the last checkpoint
    save_checkpoint(
        epochs, checkpoints_path, model, optimizer, lr_scheduler=lr_scheduler
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


@torch.no_grad()
def evaluate(model, dataloader):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets = [], []
    for spectrograms, _, speakers in dataloader:

        # Get model outputs
        model_time = time.time()
        preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_time += model_time
        epoch_preds += preds.detach().cpu().tolist()
        epoch_targets += speakers.detach().cpu().tolist()

    # Get metrics and return them
    metrics = utils.get_metrics(epoch_targets, epoch_preds, prefix="val")
    metrics["loss"] = epoch_loss
    metrics["time"] = epoch_time
    return metrics
