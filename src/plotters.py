"""plotters — lightweight plotting helpers for PSG and hypnograms.

Small utilities for visualizing polysomnography (PSG) time-series and
hypnograms used across the project. These are concise wrappers around
Matplotlib for common plotting patterns and quick interactive inspection.

Key functions
-------------
- ``plot_signal``: plot a single 1D signal with optional scaling and labels.
- ``plot_hypnogram_from_annotations``: draw a hypnogram from annotation times/values.
- ``plot_psg_with_hypnogram_arrays``: stacked PSG + hypnogram plot with a shared x-axis.

Notes
-----
- Functions call ``matplotlib.pyplot.show()`` and return ``None``. Copy the
    plotting logic if callers need direct access to ``Figure``/``Axes`` objects.
- Inputs expect NumPy arrays for time and signal values; stage labels are
    lists of strings when required.

Example
-------
>>> plot_signal(x, y, 'Time (s)', 'Voltage (uV)', title='EEG')

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_hypnogram_from_annotations(
    *,
    times: npt.NDArray[Any],
    vals: npt.NDArray[Any],
    unique_labels: Sequence[str],
    subject: Optional[str] = None,
    xlim_s: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot a hypnogram from annotation times and numeric stage values.

    Parameters
    ----------
    times
        1D array of annotation time points in seconds.
    vals
        1D array of numeric stage values (integers) corresponding to ``unique_labels``.
    unique_labels
        Ordered sequence of stage labels (e.g. ["W", "N1", "N2", "N3", "REM"]).
    subject
        Optional subject identifier used in the plot title.
    xlim_s
        Optional x-axis limits as ``(min_s, max_s)`` in seconds.

    Notes
    -----
    This function displays the figure with ``plt.show()`` and returns ``None``.
    """

    # Ensure inputs are NumPy arrays for consistent indexing and sizing
    times = np.asarray(times)
    vals = np.asarray(vals)

    # Normalize time so plot starts at zero when possible
    if times.size > 0 and times[0] != 0:
        times = times - times[0]

    # Create a compact, single-row figure for the hypnogram
    plt.figure(figsize=(10, 3))

    # Draw the hypnogram as a post-step plot to show stage durations
    plt.step(times, vals, where="post")

    # Provide a small vertical padding so ticks are not on the axes border
    plt.ylim(-0.5, len(unique_labels) - 0.5)

    # Apply x-axis limits when supplied
    if xlim_s is not None:
        plt.xlim(*xlim_s)

    # Replace numeric ticks with human-readable stage labels
    plt.yticks(list(range(len(unique_labels))), unique_labels)
    plt.xlabel("Time (s)")

    # Title: include subject id when provided
    plt.title(f"Hypnogram for subject {subject}" if subject else "Hypnogram")

    # Invert y so that 'awake' stages (usually index 0) appear at the top
    plt.gca().invert_yaxis()

    # Add a vertical grid to aid reading of stage durations
    plt.grid(True, axis="x")

    plt.tight_layout()
    plt.show()


def plot_psg_with_hypnogram_arrays(
    *,
    psg_x_data: npt.NDArray[Any],
    psg_x_label: str,
    psg_y_data: npt.NDArray[Any],
    psg_y_label: str,
    psg_y_scale: float,
    hypno_x_data: npt.NDArray[Any],
    hypno_y_data: npt.NDArray[Any],
    hypno_y_tick: Optional[Sequence[str]] = None,
    hypno_y_label: str,
    title: Optional[str] = None,
    xlim_s: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot a PSG trace with a stacked hypnogram sharing the x-axis.

    Parameters
    ----------
    psg_x_data, hypno_x_data
        1D time arrays in seconds for PSG and hypnogram annotations.
    psg_y_data
        1D PSG signal samples (will be multiplied by ``psg_y_scale``).
    psg_y_label, hypno_y_label
        Axis labels for the PSG and hypnogram plots.
    psg_y_scale
        Multiplicative scale applied to PSG signal for display.
    hypno_y_data
        Numeric stage values for the hypnogram subplot.
    hypno_y_tick
        Optional stage label strings for the hypnogram y-axis.
    title
        Optional overall figure title.
    xlim_s
        Optional x-axis limits as ``(min_s, max_s)``.
    """

    # Normalize inputs to NumPy arrays
    psg_x = np.asarray(psg_x_data)
    hypno_x = np.asarray(hypno_x_data)
    psg_y = np.asarray(psg_y_data)
    hypno_y = np.asarray(hypno_y_data)

    # Align both time axes to start at zero when data is present
    if psg_x.size > 0 and psg_x[0] != 0:
        psg_x = psg_x - psg_x[0]
    if hypno_x.size > 0 and hypno_x[0] != 0:
        hypno_x = hypno_x - hypno_x[0]

    # Create two stacked subplots that share the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    # Top: PSG waveform (scaled for visibility)
    ax1.plot(psg_x, psg_y * psg_y_scale)
    ax1.set_ylabel(psg_y_label)
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(10))
    ax1.yaxis.set_major_locator(MaxNLocator(8))

    # Bottom: hypnogram as a step plot showing stage transitions
    ax2.step(hypno_x, hypno_y, where="post")
    ax2.set_ylabel(hypno_y_label)
    ax2.set_xlabel(psg_x_label)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(10))

    # Replace numeric y-ticks with supplied labels when available
    if hypno_y_tick is not None:
        ax2.set_yticks(list(range(len(hypno_y_tick))))
        ax2.set_yticklabels(list(hypno_y_tick))

    # Optional figure title centered above subplots
    if title:
        fig.suptitle(title)

    # Apply x-limits if requested
    if xlim_s is not None:
        ax2.set_xlim(*xlim_s)

    plt.tight_layout()
    plt.show()


def plot_signal(
    *,
    x_data: npt.NDArray[Any],
    y_data: npt.NDArray[Any],
    x_label: str,
    y_label: str,
    y_scale: float = 1.0,
    title: Optional[str] = None,
    xlim_s: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot a 1D time-series signal with optional scaling and labels.

    Parameters
    ----------
    x_data
        1D array of x values (typically time in seconds).
    y_data
        1D array of signal samples.
    x_label, y_label
        Axis labels for the plot.
    y_scale
        Scale factor applied to ``y_data`` for visualization.
    title
        Optional title for the figure.
    xlim_s
        Optional x-axis limits as ``(min_s, max_s)``.
    """

    # Normalize inputs to arrays and apply requested scaling
    x = np.asarray(x_data)
    y = np.asarray(y_data) * y_scale

    # Shift time so plots commonly start at zero for readability
    if x.size > 0 and x[0] != 0:
        x = x - x[0]

    plt.figure(figsize=(10, 3))
    plt.plot(x, y)

    # Set axis labels and optional title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)

    # Apply x-limits when provided
    if xlim_s is not None:
        plt.xlim(*xlim_s)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
