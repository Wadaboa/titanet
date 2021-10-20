import random
import datetime
import os
import string

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import ConvexHull
from scipy import interpolate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def visualize_embeddings(
    embeddings,
    labels,
    labels_mapping=None,
    reduction_method="svd",
    remove_outliers=False,
    convex_hull=False,
    figsize=(12, 10),
    legend=False,
    show=True,
    save=None,
):
    """
    Plot the given embedding vectors, after reducing them to 2D
    """
    assert (
        len(embeddings.shape) == 2 and embeddings.shape[1] > 1
    ), "Wrong embeddings format/dimension"
    assert (
        len(labels.shape) == 1 and labels.shape[0] == embeddings.shape[0]
    ), "Wrong labels format/dimension"

    # Convert embeddings and labels to numpy
    embeddings, labels = to_numpy(embeddings), to_numpy(labels)

    # Compute dimesionality reduction to 2D
    if embeddings.shape[1] > 2:
        embeddings = reduce(
            embeddings, n_components=2, reduction_method=reduction_method
        )

    # Store embeddings in a dataframe and compute cluster colors
    embeddings_df = pd.DataFrame(
        np.concatenate([embeddings, np.expand_dims(labels, axis=-1)], axis=-1),
        columns=["x", "y", "l"],
    )
    embeddings_df.l = embeddings_df.l.astype(int)
    cluster_colors = {l: np.random.random(3) for l in np.unique(labels)}
    embeddings_df["c"] = embeddings_df.l.map(
        {l: tuple(c) for l, c in cluster_colors.items()}
    )

    # Plot embeddings and centroids
    fig, ax = plt.subplots(figsize=figsize)
    for l, c in cluster_colors.items():
        to_plot = embeddings_df[embeddings_df.l == l]
        label = labels_mapping[l] if labels_mapping is not None else l
        ax.scatter(to_plot.x, to_plot.y, color=c, label=f"{label}")
        ax.scatter(
            to_plot.x.mean(),
            to_plot.y.mean(),
            color=c,
            label=f"{label} (C)",
            marker="^",
        )

    # Do not represent outliers
    if remove_outliers:
        xmin_quantile = np.quantile(embeddings[:, 0], q=0.01)
        xmax_quantile = np.quantile(embeddings[:, 0], q=0.99)
        ymin_quantile = np.quantile(embeddings[:, 1], q=0.01)
        ymax_quantile = np.quantile(embeddings[:, 1], q=0.99)
        ax.set_xlim(xmin_quantile, xmax_quantile)
        ax.set_ylim(ymin_quantile, ymax_quantile)

    # Plot a shaded polygon around each cluster
    if convex_hull:
        for l, c in cluster_colors.items():
            try:
                # Get the convex hull
                points = embeddings_df[embeddings_df.l == l][["x", "y"]].values
                hull = ConvexHull(points)
                x_hull = np.append(
                    points[hull.vertices, 0], points[hull.vertices, 0][0]
                )
                y_hull = np.append(
                    points[hull.vertices, 1], points[hull.vertices, 1][0]
                )

                # Interpolate to get a smoother figure
                dist = np.sqrt(
                    (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
                )
                dist_along = np.concatenate(([0], dist.cumsum()))
                spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interp_x, interp_y = interpolate.splev(interp_d, spline)

                # Plot the smooth polygon
                ax.fill(interp_x, interp_y, "--", color=c, alpha=0.2)
            except:
                continue

    # Spawn the plot
    if legend:
        plt.legend()
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
    if show:
        plt.show()
    else:
        plt.close(fig)


def reduce(embeddings, n_components=2, reduction_method="svd", seed=42):
    """
    Applies the selected dimensionality reduction technique
    to the given input data
    """
    assert reduction_method in ("svd", "tsne"), "Unsupported reduction method"
    if reduction_method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=n_components, metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings)


def get_random_filename(length=10):
    """
    Return a random sequence of letters, to be used as unique filenames
    """
    symbols = string.ascii_lowercase
    return "".join(random.choice(symbols) for _ in range(length))


def plot_spectrogram(spectrogram, figsize=(12, 3)):
    """
    Plot the given spectrogram as an image having frequency
    on the y-axis and time on the x-axis
    """
    # If a batch of spectrograms is given, select a random one
    if len(spectrogram.shape) > 2:
        if spectrogram.size(0) > 1:
            spectrogram = spectrogram[random.randint(0, len(spectrogram) - 1)]
        else:
            spectrogram = spectrogram.squeeze(0)

    # Convert from torch to numpy
    spectrogram = to_numpy(spectrogram.squeeze(0)).astype(np.float32)

    # Plot the spectrogram
    _, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(img, ax=ax)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def play_audio(waveform, sample_rate):
    """
    Spawn an audio player in a Jupyter notebook,
    to listen to the given waveform
    """
    waveform = to_numpy(waveform)
    num_channels, _ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported")


def to_numpy(arr):
    """
    Convert the given array to the numpy format
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return None


def now():
    """
    Returns the current date and time
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_device():
    """
    Return a CUDA device, if available, or a standard CPU device otherwise
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(arr):
    """
    Flatten the given 2D array
    """
    return [item for sublist in arr for item in sublist]


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_metrics(y_true, y_pred, prefix=None):
    """
    Return a dictionary of classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    if prefix is not None:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    return metrics


def init_wandb(api_key_file, project, entity, name=None, config=None):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    return wandb.init(
        name=name,
        project=project,
        entity=entity,
        config=config,
    )
