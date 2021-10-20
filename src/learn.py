import os
import time
import math
import sys

import torch
from rich.console import Console
from rich.table import Table

import utils


# Rich console
CONSOLE = Console()


def log_step(
    current_epoch, total_epochs, current_step, total_steps, loss, time, val=False
):
    """
    Log metrics to the console after a forward pass
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split")
    table.add_column("Epoch")
    table.add_column("Step")
    table.add_column("Loss")
    table.add_column("Time")
    table.add_row(
        "Training" if not val else "Validation",
        f"{current_epoch} / {total_epochs}",
        f"{current_step} / {total_steps}",
        f"{loss}",
        f"{round(time, 2)} s",
    )
    CONSOLE.print(table)


def log_epoch(current_epoch, total_epochs, metrics, val=False):
    """
    Log metrics to the console after an epoch
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split")
    table.add_column("Epoch")
    for k in metrics:
        table.add_column(k.capitalize())
    table.add_row(
        "Training" if not val else "Validation",
        f"{current_epoch} / {total_epochs}",
        *tuple(metrics.values()),
    )
    CONSOLE.print(table)


def train_one_epoch(current_epoch, total_epochs, model, optimizer, dataloader):
    """
    Train the given model for one epoch with the given dataloader and optimizer
    """
    # Put the model in training mode
    model.train()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets = [], []
    for step, (spectrograms, _, speakers) in enumerate(dataloader):

        # Get model outputs
        model_time = time.time()
        preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Log to console
        log_step(
            current_epoch,
            total_epochs,
            step,
            len(dataloader),
            loss,
            model_time,
            val=False,
        )

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
    epoch, checkpoints_path, model, optimizer, lr_scheduler=None, wandb_run=None
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
    if wandb_run is not None:
        wandb_run.save(checkpoint_file)


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
    # For each epoch
    for epoch in range(1, epochs + 1):

        # Train for one epoch and log metrics to wandb
        train_metrics = train_one_epoch(
            epoch, epochs, model, optimizer, train_dataloader
        )
        if wandb_run is not None:
            wandb_run.log(train_metrics, step=epoch)

        # Log to console
        log_epoch(epoch, epochs, train_metrics, val=False)

        # Decay the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoints once in a while
        if checkpoints_frequency is not None and epoch % checkpoints_frequency == 0:
            save_checkpoint(
                epoch,
                checkpoints_path,
                model,
                optimizer,
                lr_scheduler=lr_scheduler,
                wandb_run=wandb_run,
            )

        # Evaluate once in a while (always evaluate at the first and last epochs)
        if epoch % val_every == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate(epoch, epochs, model, val_dataloader)
            if wandb_run is not None:
                wandb_run.log(val_metrics, step=epoch)

            # Log to console
            log_epoch(epoch, epochs, val_metrics, val=True)

    # Always save the last checkpoint
    save_checkpoint(
        epochs,
        checkpoints_path,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        wandb_run=wandb_run,
    )


@torch.no_grad()
def evaluate(current_epoch, total_epochs, model, dataloader):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets = [], []
    for step, (spectrograms, _, speakers) in enumerate(dataloader):

        # Get model outputs
        model_time = time.time()
        preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Log to console
        log_step(
            current_epoch,
            total_epochs,
            step,
            len(dataloader),
            loss,
            model_time,
            val=True,
        )

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
