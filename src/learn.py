import os
import time
import math
import sys

import torch
import wandb
from rich.console import Console
from rich.table import Table

import utils


# Rich console
CONSOLE = Console()


def log_step(
    current_epoch, total_epochs, current_step, total_steps, loss, time, prefix
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
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        f"{current_step} / {total_steps}",
        f"{loss:.2f}",
        f"{time:.2f} s",
    )
    CONSOLE.print(table)


def log_epoch(current_epoch, total_epochs, metrics, prefix):
    """
    Log metrics to the console after an epoch
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split")
    table.add_column("Epoch")
    for k in metrics:
        table.add_column(k.replace(prefix, "").replace("/", "").capitalize())
    metric_values = [f"{m:.2f}" for m in metrics.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        *tuple(metric_values),
    )
    CONSOLE.print(table)


def train_one_epoch(
    current_epoch,
    total_epochs,
    model,
    optimizer,
    dataloader,
    figures_path=None,
    wandb_run=None,
    log_console=True,
):
    """
    Train the given model for one epoch with the given dataloader and optimizer
    """
    # Put the model in training mode
    model.train()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    for step, (spectrograms, _, speakers) in enumerate(dataloader):

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Log to console
        if log_console:
            log_step(
                current_epoch,
                total_epochs,
                step,
                len(dataloader),
                loss,
                model_time,
                "train",
            )

        # Store epoch info
        epoch_loss += loss
        epoch_time += model_time
        epoch_preds += preds.detach().cpu().tolist()
        epoch_targets += speakers.detach().cpu().tolist()
        epoch_embeddings += embeddings

        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get metrics
    metrics = utils.get_metrics(epoch_targets, epoch_preds, prefix="train")
    metrics["train/loss"] = epoch_loss / len(dataloader)
    metrics["train/time"] = epoch_time

    # Log to console
    if log_console:
        log_epoch(current_epoch, total_epochs, metrics, "train")

    # Plot embeddings
    epoch_embeddings = torch.stack(epoch_embeddings)
    if figures_path is not None:
        figure_path = os.path.join(figures_path, f"epoch_{current_epoch}_train.png")
        utils.visualize_embeddings(
            epoch_embeddings,
            epoch_targets,
            show=False,
            save=figure_path,
        )
        if wandb_run is not None:
            metrics["train/embeddings"] = wandb.Image(figure_path)

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


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
    checkpoint_file = os.path.join(checkpoints_path, f"epoch_{epoch}.pth")
    torch.save(state_dict, checkpoint_file)
    if wandb_run is not None:
        wandb_run.save(checkpoint_file)


def training_loop(
    run_name,
    epochs,
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    checkpoints_path,
    val_every,
    figures_path=None,
    lr_scheduler=None,
    checkpoints_frequency=None,
    wandb_run=None,
    log_console=True,
):
    """
    Standard training loop function: train and evaluate
    after each training epoch
    """
    # Create checkpoints directory
    checkpoints_path = os.path.join(checkpoints_path, run_name)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Create figures directory
    if figures_path is not None:
        figures_path = os.path.join(figures_path, run_name)
        os.makedirs(figures_path, exist_ok=True)

    # For each epoch
    for epoch in range(1, epochs + 1):

        # Train for one epoch
        train_one_epoch(
            epoch,
            epochs,
            model,
            optimizer,
            train_dataloader,
            figures_path=figures_path,
            wandb_run=wandb_run,
            log_console=log_console,
        )

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
            evaluate(
                epoch,
                epochs,
                model,
                val_dataloader,
                "val",
                figures_path=figures_path,
                wandb_run=wandb_run,
                log_console=log_console,
            )

    # Always save the last checkpoint
    save_checkpoint(
        epochs,
        checkpoints_path,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        wandb_run=wandb_run,
    )

    # Final test
    evaluate(
        epochs + 1,
        epochs + 1,
        model,
        test_dataloader,
        "test",
        figures_path=figures_path,
        wandb_run=wandb_run,
        log_console=log_console,
    )


@torch.no_grad()
def evaluate(
    current_epoch,
    total_epochs,
    model,
    dataloader,
    prefix,
    figures_path=None,
    wandb_run=None,
    log_console=True,
):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    epoch_loss, epoch_time = 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    for step, (spectrograms, _, speakers) in enumerate(dataloader):

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(spectrograms, speakers=speakers)
        model_time = time.time() - model_time

        # Log to console
        if log_console:
            log_step(
                current_epoch,
                total_epochs,
                step,
                len(dataloader),
                loss,
                model_time,
                prefix,
            )

        # Store epoch info
        epoch_loss += loss
        epoch_time += model_time
        epoch_preds += preds.detach().cpu().tolist()
        epoch_targets += speakers.detach().cpu().tolist()
        epoch_embeddings += embeddings

    # Get metrics and return them
    metrics = utils.get_metrics(epoch_targets, epoch_preds, prefix=prefix)
    metrics[f"{prefix}/loss"] = epoch_loss / len(dataloader)
    metrics[f"{prefix}/time"] = epoch_time

    # Log to console
    if log_console:
        log_epoch(current_epoch, total_epochs, metrics, prefix)

    # Plot embeddings
    epoch_embeddings = torch.stack(epoch_embeddings)
    if figures_path is not None:
        figure_path = os.path.join(figures_path, f"epoch_{current_epoch}_{prefix}.png")
        utils.visualize_embeddings(
            epoch_embeddings,
            epoch_targets,
            show=False,
            save=figure_path,
        )
        if wandb_run is not None:
            metrics[f"{prefix}/embeddings"] = wandb.Image(figure_path)

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)
