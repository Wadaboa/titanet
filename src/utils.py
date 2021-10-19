import random
import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import ConvexHull
from scipy import interpolate


def visualize_embeddings(
    embeddings,
    labels,
    labels_mapping=None,
    reduction_method="svd",
    remove_outliers=False,
    convex_hull=False,
):
    """
    Plot the given embedding vectors, after reducing them to 2D
    """
    assert (
        isinstance(embeddings, np.ndarray)
        and len(embeddings.shape) == 2
        and embeddings.shape[1] > 1
    ), "Wrong embeddings format/dimension"
    assert (
        isinstance(labels, np.ndarray)
        and len(labels.shape) == 1
        and labels.shape[0] == embeddings.shape[0]
    ), "Wrong labels format/dimension"

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
    _, ax = plt.subplots()
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
    plt.legend()
    plt.show()


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
