import os
import time
import math
import sys
import json
from matplotlib.pyplot import figure

import torch
import torch.nn.functional as F
import wandb
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import utils


# Rich console
CONSOLE = Console()


def log_step(
    current_epoch, total_epochs, current_step, total_steps, loss, times, prefix
):
    """
    Log metrics to the console after a forward pass
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    table.add_column("STEP")
    table.add_column("LOSS")
    for time in times:
        table.add_column(f"{time.upper()} TIME")
    time_values = [f"{t:.2f}" for t in times.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        f"{current_step} / {total_steps}",
        f"{loss:.2f}",
        *tuple(time_values),
    )
    CONSOLE.print(table)


def log_epoch(current_epoch, total_epochs, metrics, prefix):
    """
    Log metrics to the console after an epoch
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    for k in metrics:
        table.add_column(k.replace(prefix, "").replace("/", "").upper())
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
    lr_scheduler=None,
    figures_path=None,
    reduction_method="svd",
    wandb_run=None,
    log_console=True,
    device="cpu",
):
    """
    Train the given model for one epoch with the given dataloader and optimizer
    """
    # Put the model in training mode
    model.train()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time, epoch_opt_time = 0, 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()
    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(
            spectrograms.to(device), speakers=speakers.to(device)
        )
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_data_time += data_time
        epoch_model_time += model_time
        epoch_embeddings += embeddings
        epoch_targets += speakers.detach().cpu().tolist()
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Stop if loss is not finite
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # Perform backpropagation
        opt_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        opt_time = time.time() - opt_time
        epoch_opt_time += opt_time

        # Log to console
        if log_console:
            times = {"model": model_time, "data": data_time, "opt": opt_time}
            log_step(
                current_epoch, total_epochs, step, len(dataloader), loss, times, "train"
            )

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    # Get metrics
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = utils.get_train_val_metrics(
            epoch_targets, epoch_preds, prefix="train"
        )
    metrics["train/total_loss"] = epoch_loss
    metrics["train/avg_loss"] = epoch_loss / len(dataloader)
    metrics["train/total_data_time"] = epoch_data_time
    metrics["train/avg_data_time"] = epoch_data_time / len(dataloader)
    metrics["train/total_model_time"] = epoch_model_time
    metrics["train/avg_model_time"] = epoch_model_time / len(dataloader)
    metrics["train/total_opt_time"] = epoch_opt_time
    metrics["train/avg_opt_time"] = epoch_opt_time / len(dataloader)
    metrics["train/lr"] = (
        lr_scheduler.get_last_lr()[0]
        if lr_scheduler is not None
        else optimizer.param_groups[0]["lr"]
    )

    # Log to console
    if log_console:
        log_epoch(current_epoch, total_epochs, metrics, "train")

    # Plot embeddings
    if figures_path is not None:
        figure_path = os.path.join(figures_path, f"epoch_{current_epoch}_train.png")
        utils.visualize_embeddings(
            torch.stack(epoch_embeddings),
            epoch_targets,
            reduction_method=reduction_method,
            show=False,
            save=figure_path,
            only_centroids=False,
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
    checkpoints_path,
    test_dataset=None,
    val_dataloader=None,
    val_every=None,
    figures_path=None,
    reduction_method="svd",
    lr_scheduler=None,
    checkpoints_frequency=None,
    wandb_run=None,
    log_console=True,
    mindcf_p_target=0.01,
    mindcf_c_fa=1,
    mindcf_c_miss=1,
    device="cpu",
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
            lr_scheduler=lr_scheduler,
            figures_path=figures_path,
            reduction_method=reduction_method,
            wandb_run=wandb_run,
            log_console=log_console,
            device=device,
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
        if (
            val_dataloader is not None
            and val_every is not None
            and (epoch % val_every == 0 or epoch == 1 or epoch == epochs)
        ):
            evaluate(
                model,
                val_dataloader,
                current_epoch=epoch,
                total_epochs=epochs,
                figures_path=figures_path,
                reduction_method=reduction_method,
                wandb_run=wandb_run,
                log_console=log_console,
                device=device,
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
    if test_dataset is not None:
        test(
            model,
            test_dataset,
            wandb_run=wandb_run,
            log_console=log_console,
            mindcf_p_target=mindcf_p_target,
            mindcf_c_fa=mindcf_c_fa,
            mindcf_c_miss=mindcf_c_miss,
            device=device,
        )


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    current_epoch=None,
    total_epochs=None,
    figures_path=None,
    figure_name=None,
    reduction_method="svd",
    wandb_run=None,
    log_console=True,
    device="cpu",
):
    """
    Evaluate the given model for one epoch with the given dataloader
    """
    # Put the model in evaluation mode
    model.eval()

    # For each batch
    step = 1
    epoch_loss, epoch_data_time, epoch_model_time = 0, 0, 0
    epoch_preds, epoch_targets, epoch_embeddings = [], [], []
    data_time = time.time()
    for spectrograms, _, speakers in dataloader:

        # Get data loading time
        data_time = time.time() - data_time

        # Get model outputs
        model_time = time.time()
        embeddings, preds, loss = model(
            spectrograms.to(device), speakers=speakers.to(device)
        )
        model_time = time.time() - model_time

        # Store epoch info
        epoch_loss += loss
        epoch_data_time += data_time
        epoch_model_time += model_time
        epoch_embeddings += embeddings
        epoch_targets += speakers.detach().cpu().tolist()
        if preds is not None:
            epoch_preds += preds.detach().cpu().tolist()

        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log to console
        if log_console:
            times = {"model": model_time, "data": data_time}
            log_step(
                current_epoch, total_epochs, step, len(dataloader), loss, times, "val"
            )

        # Increment step and re-initialize time counter
        step += 1
        data_time = time.time()

    # Get metrics and return them
    metrics = dict()
    if len(epoch_preds) > 0:
        metrics = utils.get_train_val_metrics(epoch_targets, epoch_preds, prefix="val")
    metrics[f"val/total_loss"] = epoch_loss
    metrics[f"val/avg_loss"] = epoch_loss / len(dataloader)
    metrics[f"val/total_data_time"] = epoch_data_time
    metrics[f"val/avg_data_time"] = epoch_data_time / len(dataloader)
    metrics[f"val/total_model_time"] = epoch_model_time
    metrics[f"val/avg_model_time"] = epoch_model_time / len(dataloader)

    # Log to console
    if log_console:
        log_epoch(current_epoch, total_epochs, metrics, "val")

    # Plot embeddings
    if figures_path is not None:
        if figure_name is None:
            figure_name = f"epoch_{current_epoch}_val.png"
        figure_path = os.path.join(figures_path, figure_name)
        utils.visualize_embeddings(
            torch.stack(epoch_embeddings),
            epoch_targets,
            reduction_method=reduction_method,
            show=False,
            save=figure_path,
            only_centroids=False,
        )
        if wandb_run is not None:
            metrics[f"val/embeddings"] = wandb.Image(figure_path)

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log(metrics, step=current_epoch)


@torch.no_grad()
def test(
    model,
    test_dataset,
    indices=None,
    wandb_run=None,
    log_console=True,
    mindcf_p_target=0.01,
    mindcf_c_fa=1,
    mindcf_c_miss=1,
    device="cpu",
):
    """
    Test the given model and store EER and minDCF metrics
    """
    # Put the model in evaluation mode
    model.eval()

    # Get cosine similarity scores and labels
    samples = (
        test_dataset.get_sample_pairs(indices=indices, device=device)
        if not isinstance(test_dataset, torch.utils.data.Subset)
        else test_dataset.dataset.get_sample_pairs(
            indices=test_dataset.indices, device=device
        )
    )
    scores, labels = [], []
    for s1, s2, label in tqdm(samples, desc="Building scores and labels"):
        e1, e2 = model(s1), model(s2)
        scores += [F.cosine_similarity(e1, e2).item()]
        labels += [int(label)]

    # Get test metrics (EER and minDCF)
    metrics = utils.get_test_metrics(
        scores,
        labels,
        mindcf_p_target=mindcf_p_target,
        mindcf_c_fa=mindcf_c_fa,
        mindcf_c_miss=mindcf_c_miss,
        prefix="test",
    )

    # Log to console
    if log_console:
        log_epoch(None, None, metrics, "test")

    # Log to wandb
    if wandb_run is not None:
        wandb_run.notes = json.dumps(metrics, indent=2).encode("utf-8")

    return metrics


def infer(
    model,
    utterances,
    speakers,
    dataset,
    reduction_method="svd",
    figure_path=None,
    device="cpu",
):
    """
    Compute embeddings for the given utterances and plot them
    """
    # Put the model in evaluation mode
    model.eval()

    # Compute embeddings
    all_embeddings = []
    for utterance in tqdm(utterances):
        data = dataset[utterance]
        embeddings = model(data["spectrogram"].to(device))
        all_embeddings += embeddings

    # Show embeddings
    utils.visualize_embeddings(
        torch.stack(all_embeddings),
        speakers,
        reduction_method=reduction_method,
        show=True,
        save=figure_path,
        only_centroids=False,
    )
