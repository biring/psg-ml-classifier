"""
Module for extracting features from EEG signals and building a Dataset representation.

This module provides functionality for converting raw EEG signals into feature-extracted
datasets suitable for machine learning applications. It includes tools for extracting
time-domain and frequency-domain features from EEG epochs, organizing them into labeled
datasets, and generating formatted summaries.

Key Components
--------------
FrequencyDomainFeatures : dataclass
    Immutable container for FFT-based frequency analysis results, holding power values
    aggregated into equal-width frequency bins along with associated metadata.

Dataset : dataclass
    Immutable container for feature-extracted EEG datasets with labeled epochs,
    combining both time-domain and frequency-domain features with metadata and
    conversion utilities for ML training.

Functions
---------
build_dataset(epochs, labels, sampling_freq_hz, n_bins) -> Dataset
    Main function to build a feature-extracted Dataset from raw EEG epochs.
    Processes each epoch by extracting time and frequency domain features,
    applies sleep stage label mapping, and filters to ML label set.

format_feature_dataset_summary(ds) -> str
    Generates a formatted, human-readable summary string for a Dataset,
    including array shapes, feature dimensions, extraction metadata,
    and label distribution statistics.

_generate_time_domain_features(signal) -> tuple[np.ndarray, list[str]]
    Extracts 9 time-domain statistical features (mean, std, var, min, max, rms,
    skewness, kurtosis, zero-crossing rate) from a 1D EEG signal.

_generate_frequency_domain_features(signal, fs, n_bins) -> FrequencyDomainFeatures
    Computes frequency-domain power features using FFT and aggregates power across
    equal-width frequency bins from 0 to the Nyquist frequency.

Dependencies
------------
- numpy : array operations, FFT computation, and numerical calculations
- dataclasses : immutable dataclass definitions
- constants : ML_SLEEP_STAGE_LABELS and SLEEP_STAGE_MAP for label management

Usage Example
-------------
See the ``__main__`` section for a complete example demonstrating how to load
a sleep session, extract raw EEG epochs, and create a feature-extracted Dataset
with formatted summary output.
"""

# --- library imports ---
import numpy as np

from dataclasses import dataclass
from .constants import ML_SLEEP_STAGE_LABELS, SLEEP_STAGE_MAP


@dataclass(frozen=True)
class FrequencyDomainFeatures:
    """
    Immutable container for frequency-domain features extracted from an EEG signal.

    This dataclass holds the results of FFT-based frequency analysis, including
    power values aggregated into equal-width frequency bins and associated metadata.
    The frozen dataclass ensures immutability after instantiation.

    Attributes
    ----------
    features : np.ndarray
        1D array of average power values for each frequency bin.
        Shape is (n_bins,) with dtype float64.
    bin_width_hz : float
        Width of each frequency bin in Hz (constant across all bins).
    nyquist_hz : float
        Nyquist frequency in Hz, computed as sampling_frequency / 2.
    n_bins : int
        Total number of frequency bins used during FFT aggregation.
    feature_names : tuple[str, ...]
        Tuple of feature names corresponding to each frequency bin, formatted as
        "{center_freq:.1f}Hz" where center_freq is the midpoint frequency of the bin.
    """

    # Power values aggregated across equal-width frequency bins
    features: np.ndarray
    # Width of each frequency bin in Hz
    bin_width_hz: float
    # Nyquist frequency (maximum frequency represented) in Hz
    nyquist_hz: float
    # Number of bins used to partition the frequency spectrum
    n_bins: int
    # Feature names as centered frequency +/- bin_width_hz/2
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    """
    Immutable container for feature-extracted EEG dataset with labeled epochs.

    This class holds processed EEG epochs where each epoch contains both time-domain
    and frequency-domain features, along with corresponding labels and extraction metadata.
    The frozen dataclass ensures immutability after instantiation.

    Attributes
    ----------
    x_epochs : tuple[tuple[np.ndarray, ...], int]
        Tuple of processed epochs, where each epoch is a 2-tuple:
        - [0]: tuple of feature arrays (e.g., (time_domain_features, frequency_domain_features))
        - [1]: integer label index corresponding to a label in y_labels.
    y_labels : tuple[str, ...]
        Tuple of label names (e.g., ("Wake", "N1", "N2", "N3", "REM")).
        Index position in this tuple corresponds to label integer values in epochs.
    stats : dict[str, float]
        Dictionary of metadata captured during feature extraction, such as sampling
        frequency, Nyquist frequency, number of bins, and feature dimensions.
    """

    x_epochs: tuple[tuple[np.ndarray, ...], int]
    y_labels: tuple[str, ...]
    stats: dict[str, float]

    def __len__(self) -> int:
        """Return the total number of epochs in the dataset."""
        return len(self.x_epochs)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert epochs and labels to stacked NumPy arrays suitable for ML training.

        Combines all feature arrays (time-domain and frequency-domain) from each epoch
        into a single flattened feature vector, then stacks across all epochs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - X : np.ndarray of shape (n_epochs, n_features)
              Stacked feature vectors from all epochs (time + frequency features flattened).
            - y : np.ndarray of shape (n_epochs,) with dtype int64
              Label indices corresponding to each epoch.

        Raises
        ------
        ValueError
            If the dataset contains no epochs.
        """
        if len(self.x_epochs) == 0:
            raise ValueError("No epochs available.")

        # Collect feature vectors and labels for all epochs
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for (time_feat, freq_feat), label in self.x_epochs:
            # Flatten time-domain features and concatenate with frequency-domain features
            combined: np.ndarray = np.concatenate(
                [time_feat.flatten(), np.asarray(freq_feat)]
            )
            X_list.append(combined)
            y_list.append(label)

        # Stack all feature vectors into a 2D array (n_epochs, n_features)
        X: np.ndarray = np.stack(X_list, axis=0)
        # Convert labels to 1D integer array
        y: np.ndarray = np.asarray(y_list, dtype=np.int64)

        return X, y


def _generate_time_domain_features(signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract time-domain statistical features from a 1D EEG signal.

    Computes multiple statistical measures that characterize the signal's
    amplitude distribution and temporal behavior, useful for sleep stage
    classification and signal analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input EEG signal as a 1D array (e.g., 30 seconds of data).

    Returns
    -------
    np.ndarray
        1D array of 9 time-domain features in the following order:
        [mean, std, variance, min, max, rms, skewness, kurtosis, zcr]

    Notes
    -----
    - All features are computed directly on the input signal without normalization.
    - Skewness and kurtosis use a small epsilon (1e-8) to avoid division by zero.
    - Zero-crossing rate is normalized by signal length.
    """
    features: list[np.ndarray] = []
    feature_names: list[str] = []

    # Mean: average amplitude across the signal
    features.append(np.mean(signal, axis=0))
    feature_names.append("mean")

    # Standard deviation: measure of amplitude spread around the mean
    features.append(np.std(signal, axis=0))
    feature_names.append("standard_deviation")

    # Variance: squared standard deviation
    features.append(np.var(signal, axis=0))
    feature_names.append("variance")

    # Minimum: lowest amplitude value in the signal
    features.append(np.min(signal, axis=0))
    feature_names.append("minimum")

    # Maximum: highest amplitude value in the signal
    features.append(np.max(signal, axis=0))
    feature_names.append("maximum")

    # Root mean square: measure of signal energy
    features.append(np.sqrt(np.mean(signal**2, axis=0)))
    feature_names.append("root_mean_square")

    # Skewness: asymmetry of the amplitude distribution (normalized by std^3)
    features.append(
        np.mean((signal - np.mean(signal, axis=0)) ** 3, axis=0)
        / (np.std(signal, axis=0) ** 3 + 1e-8)
    )
    feature_names.append("skewness")

    # Kurtosis: tail heaviness of the amplitude distribution (normalized by std^4)
    features.append(
        np.mean((signal - np.mean(signal, axis=0)) ** 4, axis=0)
        / (np.std(signal, axis=0) ** 4 + 1e-8)
    )
    feature_names.append("kurtosis")

    # Zero-crossing rate: proportion of times the signal crosses zero
    features.append(np.mean(np.diff(np.sign(signal), axis=0) != 0, axis=0))
    feature_names.append("zero_crossing_rate")

    return np.array(features), feature_names


def _generate_frequency_domain_features(
    signal: np.ndarray, fs: float, n_bins: int
) -> FrequencyDomainFeatures:
    """
    Compute frequency-domain power features using equal-width bins.

    Applies FFT to the input signal and aggregates power across equal-width
    frequency bins from 0 to the Nyquist frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input EEG signal as a 1D array (e.g., 30 seconds of data).
    fs : float
        Sampling frequency in Hz.
    n_bins : int
        Number of equal-width frequency bins between 0 and Nyquist frequency.

    Returns
    -------
    FrequencyDomainFeatures
        Dataclass containing:
        - bin_powers: np.ndarray of average power per bin.
        - bin_width_hz: float bin width in Hz.
        - nyquist_hz: float Nyquist frequency in Hz.
        - n_bins: int number of bins.

    Raises
    ------
    ValueError
        If n_bins is non-positive, fs is non-positive, signal is not 1D,
        n_bins exceeds Nyquist frequency, or signal is empty.
    """
    # Validate input parameters
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    if n_bins > fs / 2:
        raise ValueError("n_bins cannot exceed Nyquist frequency (fs/2).")

    # Convert to numpy array and get signal length
    x: np.ndarray = np.asarray(signal)
    N: int = len(x)

    if N == 0:
        raise ValueError("Input signal is empty.")

    # Compute FFT and extract magnitude-squared power spectrum
    fft_vals: np.ndarray = np.fft.rfft(x)
    power: np.ndarray = np.abs(fft_vals) ** 2

    # Compute positive frequency components (0 to Nyquist)
    freqs: np.ndarray = np.fft.rfftfreq(N, d=1 / fs)
    nyquist: float = fs / 2

    # Calculate bin width in Hz
    bin_width: float = nyquist / n_bins

    # Map each frequency to its bin index and clip to valid range [0, n_bins-1]
    bin_idx: np.ndarray = np.floor(freqs / bin_width).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Aggregate power values per bin: compute sum and count of frequencies per bin
    power_sum: np.ndarray = np.bincount(bin_idx, weights=power, minlength=n_bins)
    power_cnt: np.ndarray = np.bincount(bin_idx, minlength=n_bins)

    # Compute mean power per bin, handling empty bins with zero
    bin_powers: np.ndarray = np.divide(
        power_sum,
        power_cnt,
        out=np.zeros(n_bins, dtype=np.float64),
        where=power_cnt > 0,
    )

    # Generate feature names as centered frequency +/- bin_width/2
    feature_names: tuple[str, ...] = tuple(
        f"{(i + 0.5) * bin_width:.1f}Hz" for i in range(n_bins)
    )

    return FrequencyDomainFeatures(
        features=bin_powers,
        bin_width_hz=bin_width,
        nyquist_hz=nyquist,
        n_bins=n_bins,
        feature_names=feature_names,
    )


def build_dataset(
    *,
    epochs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    sampling_freq_hz: float,
    n_bins: int,
) -> Dataset:
    """
    Build a feature-extracted Dataset from raw EEG epochs.

    Processes each raw epoch by extracting time-domain and frequency-domain features,
    then returns a Dataset object suitable for machine learning training.

    Parameters
    ----------
    epochs : tuple[np.ndarray, ...]
        Tuple of epoch objects, each with `x` (1D signal array) and `label` (int) attributes.
    labels : tuple[str, ...]
        Tuple of label names, e.g., ("Wake", "N1", "N2", "N3", "REM").
    sampling_freq_hz : float
        Sampling frequency of the EEG signal in Hz.
    n_bins : int
        Number of frequency bins for the frequency-domain feature extraction.

    Returns
    -------
    Dataset
        A Dataset object containing processed epochs, label names, and extraction metadata.

    Raises
    ------
    ValueError
        If epochs is empty, sampling frequency is non-positive, n_bins is non-positive,
        or if any epoch has invalid shape/content.
    """
    # Validate input parameters
    if not epochs:
        raise ValueError("No epochs provided.")
    if sampling_freq_hz <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if n_bins <= 0:
        raise ValueError("Number of frequency bins must be a positive integer.")

    # Dictionary to store metadata about the dataset and feature extraction process
    stats: dict[str, float] = {}

    # List to accumulate features for all epochs
    processed_epochs: list[tuple[tuple[np.ndarray, np.ndarray], int]] = []

    for count, epoch in enumerate(epochs):
        # Extract raw signal and label from current epoch
        raw_x_data: np.ndarray = epoch.x
        label: int = epoch.label

        # Validate signal dimensions and content
        if raw_x_data.ndim != 1:
            raise ValueError(f"Expected 1D input signal, got {raw_x_data.ndim}D array.")
        if len(raw_x_data) == 0:
            raise ValueError("Input signal is empty.")

        # Generate time-domain features from the raw signal
        time_features: np.ndarray = np.array([])
        time_feature_names: list[str] = []
        time_features, time_feature_names = _generate_time_domain_features(
            signal=raw_x_data
        )

        # Generate frequency-domain features from the raw signal
        frequency_feature: FrequencyDomainFeatures = (
            _generate_frequency_domain_features(
                signal=raw_x_data,
                fs=sampling_freq_hz,
                n_bins=n_bins,
            )
        )

        # Combine time and frequency features for this epoch
        feature_x_data: tuple[np.ndarray, np.ndarray] = (
            time_features,
            frequency_feature.features,
        )

        # Validate epoch label index
        if not (0 <= label < len(labels)):
            raise ValueError(f"Epoch label index out of range: {label}")

        # Raw label name from provided labels tuple
        raw_label_name: str = labels[label]

        # Map raw label name using SLEEP_STAGE_MAP if available, otherwise use raw
        mapped_label_name: str = SLEEP_STAGE_MAP.get(raw_label_name, raw_label_name)

        # Keep only epochs whose mapped label name is in the ML label set
        if mapped_label_name not in ML_SLEEP_STAGE_LABELS:
            # skip this epoch (not part of ML labels)
            continue

        # Remap label to index within the ML label list
        remapped_label: int = int(ML_SLEEP_STAGE_LABELS.index(mapped_label_name))
        processed_epochs.append((feature_x_data, remapped_label))

        # Store frequency feature metadata from the first epoch (same for all epochs)
        if count == 0:
            stats["sampling_freq_hz"] = sampling_freq_hz
            stats["nyquist_hz"] = frequency_feature.nyquist_hz
            stats["n_bins"] = float(frequency_feature.n_bins)
            stats["bin_width_hz"] = frequency_feature.bin_width_hz
            stats["time_feature_names"] = (time_feature_names,)
            stats["frequency_feature_names"] = (frequency_feature.feature_names,)

    # Validate that at least one epoch was processed
    if not processed_epochs:
        raise ValueError("No epochs were processed. Check the input data.")

    # Record total number of processed epochs
    stats["total_epochs"] = float(len(processed_epochs))

    # Store feature dimensions from the first processed epoch
    if processed_epochs:
        stats["time_feature_dim"] = float(processed_epochs[0][0][0].size)
        stats["frequency_feature_dim"] = float(processed_epochs[0][0][1].size)

    # Use ML label set as the dataset labels (only epochs from that set were kept)
    return Dataset(
        x_epochs=tuple(processed_epochs),
        y_labels=tuple(ML_SLEEP_STAGE_LABELS),
        stats=stats,
    )


def format_feature_dataset_summary(ds: Dataset) -> str:
    """
    Generate a complete, formatted summary string for a feature-extracted Dataset.

    This function provides a human-readable overview of a feature-extracted dataset,
    including dataset size, feature dimensions, array shapes, extraction metadata,
    and label distribution statistics. It mirrors the functionality of
    ``format_sleep_dataset_summary`` in the ``sleep_session`` module.

    Parameters
    ----------
    ds : Dataset
        Feature-extracted dataset object returned by ``build_dataset``.

    Returns
    -------
    str
        Formatted multi-line summary string with sections for basic statistics,
        feature dimensions, array shapes, metadata, and label distribution.
        Suitable for console output or logging.

    Raises
    ------
    ValueError
        If the dataset contains no epochs, making it impossible to generate a summary.

    Examples
    --------
    >>> dataset = build_dataset(epochs=..., labels=..., sampling_freq_hz=100, n_bins=25)
    >>> print(format_feature_dataset_summary(dataset))
    """

    # Get the total number of epochs in the dataset
    n_epochs: int = len(ds.x_epochs)

    # Build stacked arrays once to extract shapes and compute label distribution
    if n_epochs:
        # Convert epoch tuples to flattened feature arrays and labels
        X: np.ndarray
        y: np.ndarray
        X, y = ds.to_arrays()
        # Extract time-domain feature shape from first epoch
        time_dim: tuple[int, ...] = ds.x_epochs[0][0][0].shape
        # Extract frequency-domain feature shape from first epoch
        freq_dim: tuple[int, ...] = ds.x_epochs[0][0][1].shape
    else:
        raise ValueError("Dataset contains no epochs, cannot generate summary.")

    # Initialize the summary lines with core dataset information
    lines: list[str] = [
        "",
        "-" * 40,
        "Array Shapes",
        "-" * 40,
        f"X shape : {X.shape}",
        f"y shape : {y.shape}",
        "",
        "Feature Summary",
        "-" * 40,
        f"Time-domain feature  : {time_dim}",
        f"Freq-domain feature  : {freq_dim}",
        f"y_labels             : {ds.y_labels}",
    ]

    # Append optional metadata captured during feature extraction
    if ds.stats:
        lines.extend(["", "Metadata (stats)", "-" * 40])
        # Iterate through stats dictionary in sorted key order for consistent output
        for k in ds.stats.keys():
            lines.append(f"{k:<30} : {ds.stats[k]}")

    # Append label distribution statistics
    lines.extend(["", "Label Distribution", "-" * 40])
    if y.size > 0:
        # Compute unique labels and their counts using NumPy
        unique: np.ndarray
        counts: np.ndarray
        unique, counts = np.unique(y, return_counts=True)
        # Calculate total epochs for percentage computation
        total: float = float(y.size)

        # Format each label with its count and percentage of total
        for u, c in zip(unique, counts):
            # Convert label index to integer for lookup
            u_int: int = int(u)
            # Map label index to label name, or use fallback format
            label_name: str = (
                ds.y_labels[u_int]
                if 0 <= u_int < len(ds.y_labels)
                else f"label_{u_int}"
            )
            # Append formatted line with label name, count, and percentage
            lines.append(
                f"{label_name:<20} : {int(c):>5} epochs ({(c/total)*100:6.2f}%)"
            )
    else:
        raise ValueError(
            "No labels found in dataset, cannot compute label distribution."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Import required modules for loading sleep session data
    from . import folders as folder
    from . import sleep_session as session

    # --- load a sample sleep session ---
    # Select a sample EEG file from the sleep data directory
    sample_file_index = 0  # adjust as needed to select a different file
    sample_eeg_file = folder.get_edf_files_in_sleep_data()[sample_file_index]
    # Select the corresponding hypnogram file with sleep stage labels
    sample_hyp_file = folder.get_hypnogram_files_in_sleep_data()[sample_file_index]
    # Create a SleepSession object by pairing the EEG and hypnogram files
    sleep_session = session.SleepSession(sample_eeg_file, sample_hyp_file)

    # --- build dataset (single channel + hypnogram) for ML training ---
    # Extract raw sleep dataset from a specific EEG channel with 30-second epochs
    ds = sleep_session.get_sleep_dataset(
        channel="EEG Fpz-Cz",  # Select the Fpz-Cz electrode channel
        epoch_len_s=30,  # Use 30-second epochs as per standard sleep analysis
        filter_cfg=session.FilterConfig(
            enabled=False,  # Disable filtering for this example (use raw signal)
        ),
        remap_cfg=session.MappingConfig(
            enabled=False,  # Disable label remapping (use original sleep stages)
        ),
    )

    # Build feature-extracted dataset with time and frequency domain features
    dataset: Dataset = build_dataset(
        epochs=ds.epochs,  # Raw EEG epochs extracted from the sleep session
        labels=ds.unique_labels,  # Sleep stage labels (Wake, N1, N2, N3, REM)
        sampling_freq_hz=ds.sampling_rate,  # Sampling frequency of the EEG signal
        n_bins=25,  # Number of frequency bins for FFT-based feature extraction
    )
    # Print a formatted summary of the feature-extracted dataset
    print(format_feature_dataset_summary(dataset))
