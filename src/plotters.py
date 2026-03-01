"""
Lightweight plotting helpers for PSG signals and hypnograms.

This module provides convenience functions to visualize polysomnography (PSG)
signals and associated hypnograms. The helpers produce Matplotlib figures
and display them interactively; they are small wrappers around common
plotting patterns used across the project.

Notes
-----
- All plotting functions call ``plt.show()`` and therefore display figures
    directly; they return ``None``. Callers who need figure objects can copy
    the logic here and remove the final ``plt.show()``.
- Inputs expect NumPy arrays for time and signal values; stage labels are
    supplied as lists of strings where required.

Example
-------
>>> plot_signal(x_data, y_data, 'Time (s)', 'Voltage (uV)', title='EEG')

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_hypnogram_from_annotations(
    *,
    times: np.ndarray,
    vals: np.ndarray,
    unique_labels: list[str],
    subject: str | None = None,
    xlim_s: tuple[float, float] | None = None,
) -> None:
    """
    Plot a hypnogram from annotation times and values.

    Parameters
    ----------
    times : np.ndarray
        Time points in seconds for each sleep stage annotation.
    vals : np.ndarray
        Numeric values corresponding to each sleep stage (0 to len(unique_labels)-1).
    unique_labels : list[str]
        List of sleep stage labels in order (e.g., ["W", "N1", "N2", "N3", "REM"]).
    subject : str, optional
        Subject identifier for the plot title. If None, a generic title is used.
    xlim_s : tuple[float, float], optional
        X-axis limits (min, max) in seconds.
    """
    # Create figure with specified size
    plt.figure(figsize=(10, 3))

    # Plot as step function to show stage transitions
    plt.step(times, vals, where="post")

    # Set y-axis limits with padding
    plt.ylim(-0.5, len(unique_labels) - 0.5)

    # Set x-axis limits if provided
    if xlim_s is not None:
        plt.xlim(*xlim_s)

    # Map numeric values to stage labels on y-axis
    plt.yticks(list(range(len(unique_labels))), unique_labels)

    # Label x-axis
    plt.xlabel("Time (s)")

    # Set title with or without subject identifier
    if subject is not None:
        plt.title(f"Hypnogram for subject {subject}")
    else:
        plt.title("Hypnogram")

    # Invert y-axis so awake (W) is at top
    plt.gca().invert_yaxis()

    # Add grid for easier reading
    plt.grid(True, axis="x")

    # Optimize layout and display
    plt.tight_layout()
    plt.show()


def plot_psg_with_hypnogram_arrays(
    *,
    time_data: np.ndarray,
    time_label: str,
    psg_data: np.ndarray,
    psg_label: str,
    hypno_time: np.ndarray,
    hypno_data: np.ndarray,
    hypno_label: str,
    title: str | None = None,
    hypno_tick_labels: list[str] | None = None,
    xlim_s: tuple[float, float] | None = None,
) -> None:
    """
    Plot PSG signal and hypnogram on separate subplots with shared x-axis.

    Parameters
    ----------
    time_data : np.ndarray
        Time points for PSG data in seconds.
    time_label : str
        Label for the time axis.
    psg_data : np.ndarray
        PSG signal values.
    psg_label : str
        Label for the PSG signal y-axis.
    hypno_time : np.ndarray
        Time points for hypnogram annotations in seconds.
    hypno_data : np.ndarray
        Numeric sleep stage values for hypnogram.
    hypno_label : str
        Label for the hypnogram y-axis.
    title : str, optional
        Figure title. If None, no title is displayed.
    hypno_tick_labels : list[str], optional
        Sleep stage labels for hypnogram y-axis ticks.
    xlim_s : tuple[float, float], optional
        X-axis limits (min, max) in seconds.
    """
    # Create figure with two vertically stacked subplots sharing the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot PSG signal on top subplot
    ax1.plot(time_data, psg_data)
    ax1.set_ylabel(psg_label)
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(10))
    ax1.yaxis.set_major_locator(MaxNLocator(8))

    # Plot hypnogram as step function on bottom subplot
    ax2.step(hypno_time, hypno_data, where="post")
    ax2.set_ylabel(hypno_label)
    ax2.set_xlabel(time_label)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(10))

    # Set custom y-axis labels for hypnogram if provided
    if hypno_tick_labels is not None:
        ax2.set_yticks(list(range(len(hypno_tick_labels))))
        ax2.set_yticklabels(list(hypno_tick_labels))

    # Add figure title if provided
    if title:
        fig.suptitle(title)

    # Set x-axis limits if provided
    if xlim_s is not None:
        ax2.set_xlim(*xlim_s)

    # Optimize layout and display
    plt.tight_layout()
    plt.show()


def plot_signal(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_label: str,
    y_label: str,
    y_scale: float = 1.0,
    title: str | None = None,
    xlim_s: tuple[float, float] | None = None,
) -> None:
    """
    Plot a 1D signal with optional scaling and title.

    Parameters
    ----------
    x_data : np.ndarray
        X-axis data points.
    y_data : np.ndarray
        Y-axis data points.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    y_scale : float, optional
        Multiplicative scaling factor applied to y_data. Default is 1.0.
    title : str, optional
        Figure title. If None, no title is displayed.
    """
    # Convert inputs to numpy arrays for consistency
    x = np.asarray(x_data)
    y = np.asarray(y_data) * y_scale

    # Create figure with specified size
    plt.figure(figsize=(10, 3))

    # Plot the signal
    plt.plot(x, y)

    # Set axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add title if provided
    if title:
        plt.title(title)

    # Set x-axis limits if provided
    if xlim_s is not None:
        plt.xlim(*xlim_s)

    # Enable grid for easier reading
    plt.grid(True)

    # Optimize layout and display
    plt.tight_layout()
    plt.show()
