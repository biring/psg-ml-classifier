"""
Signal processing helpers.

This module provides zero-phase band-pass and notch filters and a tapering
helper used throughout the project. Functions validate their inputs and
return arrays with the same shape as the input. The implementation uses
``scipy.signal`` for filter design and zero-phase filtering and ``numpy``
for array handling.

Public API
----------
- ``FilterConfig``: dataclass describing default filter parameters.
- ``bandpass_filter``: zero-phase Butterworth band-pass filter.
- ``notch``: zero-phase IIR notch filter.
- ``apply_taper``: apply a Hann or Hamming taper to 1D or 2D signals.

Notes
-----
All filters use zero-phase forward-backward filtering to avoid phase
distortion. Caller is responsible for providing appropriate sampling
frequencies and cutoff values.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    l_freq: float = 0.001
    h_freq: float = 50.0
    order: int = 1


def bandpass_filter(
    *,
    signal: np.ndarray,
    sampling_frequency: float,
    l_freq: float,
    h_freq: float,
    order: int,
) -> np.ndarray:
    """
    Apply a band-pass filter (zero-phase) using Butterworth IIR + sosfiltfilt.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array (1D).
    sampling_frequency : float
        Sampling frequency in Hz.
    l_freq : float
        Low cutoff frequency in Hz.
    h_freq : float
        High cutoff frequency in Hz.
    order : int
        Filter order (must be positive).

    Returns
    -------
    np.ndarray
        Filtered signal with same shape as input.

    Raises
    ------
    ValueError
        If signal is not 1D, sfreq <= 0, cutoff frequencies are invalid, or order <= 0.
    """
    # Convert input to float array
    x = np.asarray(signal, dtype=float)

    # Validate input is 1D
    if x.ndim != 1:
        raise ValueError("bandpass_filter() expects a 1D signal array.")

    # Validate and convert sampling frequency
    sampling_frequency = float(sampling_frequency)
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be > 0.")

    # Calculate Nyquist frequency
    nyq: float = 0.5 * sampling_frequency

    # Validate and convert cutoff frequencies
    l_freq = float(l_freq)
    h_freq = float(h_freq)

    if not (0.0 < l_freq < h_freq < nyq):
        raise ValueError(
            f"Invalid cutoffs: need 0 < l_freq < h_freq < Nyquist({nyq}). "
            f"Got l_freq={l_freq}, h_freq={h_freq}."
        )

    # Validate filter order
    if int(order) <= 0:
        raise ValueError("order must be a positive integer.")

    # Design Butterworth filter in second-order sections form
    sos: np.ndarray = butter(
        N=int(order),
        Wn=[l_freq / nyq, h_freq / nyq],
        btype="bandpass",
        output="sos",
    )

    # Apply zero-phase filtering using second-order sections
    return sosfiltfilt(sos, x)


def notch(
    *,
    signal: np.ndarray,
    sampling_frequency: float,
    freq_notch: float,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply a notch filter (zero-phase) using IIR notch filter + filtfilt.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array (1D).
    sampling_frequency : float
        Sampling frequency in Hz.
    freq_notch : float
        Notch frequency in Hz (frequency to remove).
    q : float, optional
        Quality factor controlling bandwidth. Higher Q = narrower notch.
        Default is 30.0.

    Returns
    -------
    np.ndarray
        Filtered signal with same shape as input.

    Raises
    ------
    ValueError
        If signal is not 1D, sampling_frequency <= 0, freq_notch is invalid, or q <= 0.
    """
    # Convert input to float array
    x = np.asarray(signal, dtype=float)

    # Validate input is 1D
    if x.ndim != 1:
        raise ValueError("notch() expects a 1D signal array.")

    # Validate and convert sampling frequency
    sampling_frequency = float(sampling_frequency)
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be > 0.")

    # Calculate Nyquist frequency
    nyq: float = 0.5 * sampling_frequency

    # Validate and convert notch frequency
    freq_notch = float(freq_notch)
    if not (0.0 < freq_notch < nyq):
        raise ValueError(
            f"Invalid notch freq: need 0 < freq_notch < Nyquist({nyq}). Got {freq_notch}."
        )

    # Validate and convert quality factor
    q = float(q)
    if q <= 0:
        raise ValueError("q must be > 0.")

    # Design notch filter and apply zero-phase filtering
    b, a = iirnotch(w0=freq_notch / nyq, Q=q)
    return filtfilt(b, a, x)


def apply_taper(*, x: np.ndarray, kind: str = "hann") -> np.ndarray:
    """
    Apply a taper window to a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal. Can be 1D (single epoch) or 2D (n_epochs, n_samples).
    kind : str, optional
        Type of taper window. Either "hann" or "hamming". Default is "hann".

    Returns
    -------
    np.ndarray
        Tapered signal with same shape as input.

    Raises
    ------
    ValueError
        If input is not 1D or 2D, or if kind is not "hann" or "hamming".
    """
    # Convert input to float array
    x = np.asarray(x, dtype=float)

    # Determine window length based on array dimensionality
    if x.ndim == 1:
        n = x.shape[0]
    elif x.ndim == 2:
        n = x.shape[1]
    else:
        raise ValueError("apply_taper expects 1D or 2D array.")

    # Create taper window based on kind parameter
    if kind == "hann":
        w = np.hanning(n)
    elif kind == "hamming":
        w = np.hamming(n)
    else:
        raise ValueError("kind must be 'hann' or 'hamming'.")

    # Apply window: broadcast for 2D case using None (new axis)
    if x.ndim == 1:
        return x * w
    else:  # 2D case: multiply each epoch by window
        return x * w[None, :]
