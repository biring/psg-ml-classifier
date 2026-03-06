"""
Session helpers for combining PSG and hypnogram data.

This module provides integrated access to polysomnography (PSG) signals and
sleep stage annotations (hypnogram), with support for signal filtering,
label remapping, and epoch-based dataset construction.

Key Classes
-----------
- ``SleepSession``: Main wrapper combining PsgReader and HypnogramReader.
    Provides methods to retrieve filtered channel data, remapped hypnogram data,
    synchronized plots, and epoch-aligned datasets for machine learning.
- ``SleepDataPlot``: Immutable container for synchronized PSG and hypnogram
    plot data, ready for visualization.
- ``Epoch``: Single fixed-duration epoch with signal samples and sleep stage label.
- ``Epochs``: Container for an epoch collection with shared metadata and statistics.

Key Functions
-------------
- ``format_sleep_dataset_summary``: Generate formatted summary statistics for
    an Epochs dataset.

Features
--------
- **Channel filtering**: Optional bandpass filtering of PSG signals via FilterConfig.
- **Label remapping**: Optional remapping of hypnogram sleep stage labels via MappingConfig.
- **Temporal alignment**: Automatic synchronization of PSG and hypnogram data to
    absolute time (seconds since epoch).
- **Epoch construction**: Uniform grid epoch extraction with hypnogram label assignment
    and boundary validation.
- **Data validation**: Defensive input validation and comprehensive error reporting.

The module emphasizes explicit type hints, immutable dataclasses, and safe
data handling for use in visualization pipelines and machine learning workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np

from .psg_reader import PsgReader  # PSG reading module
from .psg_reader import ChannelData  # dataclass for channel data

from .hypnogram_reader import HypnogramReader  # hypnogram reading module
from .hypnogram_reader import HypnogramData  # dataclass for hypnogram data
from .hypnogram_reader import MappingConfig  # dataclass for hypnogram remapping config

from .signal_processing import bandpass_filter  # filtering helper
from .signal_processing import FilterConfig  # dataclass for filter configuration


@dataclass(frozen=True)
class SleepDataPlot:
    """
    Immutable container for synchronized PSG and hypnogram plot data.

    This dataclass holds all necessary data and metadata for visualizing polysomnography (PSG)
    signals alongside sleep stage annotations (hypnogram). It combines filtered/processed PSG
    channel data with aligned hypnogram segments, labels, and scaling information in a format
    ready for direct consumption by plotting utilities.

    The dataclass is frozen (immutable) to ensure plot data cannot be accidentally modified
    after creation, supporting safe use in visualization pipelines and data analysis workflows.

    Attributes
    ----------
    psg_x_data : np.ndarray
        Shape (N,), dtype float64. Absolute time values (seconds since epoch) for each PSG
        sample point. Synchronized with ``psg_y_data`` for x-axis positioning in plots.
    psg_x_label : str
        Label string for the PSG x-axis (e.g., "Time (s)" or "Absolute Time (s since epoch)").
    psg_y_data : np.ndarray
        Shape (N,), dtype float32. PSG/EEG signal amplitude values in microvolts (or native
        units) for each sample. May be raw or filtered depending on filter configuration.
    psg_y_label : str
        Label string for the PSG y-axis (e.g., "EEG Fpz-Cz" or "Amplitude (µV)").
    psg_y_scale : float
        Scale factor for PSG y-axis values (typically 1.0 for identity scaling, or a channel-
        specific scaling factor from the EDF header). Used by plotting routines to adjust
        amplitude display.
    hypno_x_data : np.ndarray
        Shape (M,), dtype float64. Absolute time values (seconds since epoch) for each
        hypnogram annotation point. Aligned with PSG time axis for synchronized visualization.
    hypno_x_label : str
        Label string for the hypnogram x-axis (e.g., "Time (s)").
    hypno_y_data : np.ndarray
        Shape (M,), dtype int. Integer sleep stage identifiers (0=Wake, 1=N1, 2=N2, 3=N3,
        4=REM, etc.) for each hypnogram annotation. Maps to label names via
        ``hypno_y_tick_labels``.
    hypno_y_label : str
        Label string for the hypnogram y-axis (e.g., "Sleep Stage").
    hypno_y_tick_labels : tuple[str, ...]
        Tuple of sleep stage label names (e.g., ("Wake", "N1", "N2", "N3", "REM")) in order
        corresponding to integer indices in ``hypno_y_data``. Used to format y-axis tick
        labels in plots.
    title : str | None, optional
        Optional title string for the combined plot. If None, no title is displayed.
        Default is None.

    Notes
    -----
    - PSG and hypnogram data are pre-synchronized to absolute time (seconds since epoch)
      by ``SleepSession.get_sleep_data_plot()``, making them suitable for direct overlay
      in visualization.
    - Both x-axis arrays should have compatible time ranges; hypnogram data is typically
      a subset of the PSG time window based on hypnogram coverage.
    - The frozen dataclass design prevents accidental field mutations, supporting defensive
      programming and functional data pipelines.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple SleepDataPlot with sample data
    >>> psg_times = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
    >>> psg_signal = np.array([1.0, 1.5, 2.0, 1.5, 1.0], dtype=np.float32)
    >>> hypno_times = np.array([0.0, 30.0, 60.0])
    >>> hypno_stages = np.array([0, 1, 2])  # Wake, N1, N2
    >>> labels = ("Wake", "N1", "N2", "N3", "REM")
    >>> plot_data = SleepDataPlot(
    ...     psg_x_data=psg_times,
    ...     psg_x_label="Time (s)",
    ...     psg_y_data=psg_signal,
    ...     psg_y_label="EEG Fpz-Cz (µV)",
    ...     psg_y_scale=1.0,
    ...     hypno_x_data=hypno_times,
    ...     hypno_x_label="Time (s)",
    ...     hypno_y_data=hypno_stages,
    ...     hypno_y_label="Sleep Stage",
    ...     hypno_y_tick_labels=labels,
    ...     title="Sleep Session Plot",
    ... )
    >>> print(plot_data.psg_x_data.shape, plot_data.hypno_y_data.shape)
    (5,) (3,)

    See Also
    --------
    SleepSession.get_sleep_data_plot : Method to generate SleepDataPlot instances.
    Epochs : Container for epoch-based sleep dataset.
    """

    # ---- PSG (polysomnography) data ----
    psg_x_data: (
        np.ndarray
    )  # shape: (N,), dtype: float64; absolute time values for PSG samples
    psg_x_label: str  # label for x-axis (e.g., "Time (s)")
    psg_y_data: (
        np.ndarray
    )  # shape: (N,), dtype: float32; PSG signal amplitude in microvolts
    psg_y_label: str  # label for y-axis (e.g., "EEG Fpz-Cz (µV)")
    psg_y_scale: float  # scale factor applied to y-axis values (typically 1.0)

    # ---- Hypnogram (sleep stage annotation) data ----
    hypno_x_data: (
        np.ndarray
    )  # shape: (M,), dtype: float64; absolute time values for hypnogram points
    hypno_x_label: str  # label for x-axis (e.g., "Time (s)")
    hypno_y_data: (
        np.ndarray
    )  # shape: (M,), dtype: int; integer sleep stage labels (0, 1, 2, ...)
    hypno_y_label: str  # label for y-axis (e.g., "Sleep Stage")
    hypno_y_tick_labels: tuple[
        str, ...
    ]  # sleep stage names (e.g., ("Wake", "N1", "N2", "N3", "REM"))

    # ---- Plot metadata ----
    title: str | None = None  # optional title for the combined PSG + hypnogram plot


@dataclass(frozen=True)
class Epoch:
    """
    A single fixed-duration epoch containing PSG/EEG signal samples and sleep stage label.

    This immutable dataclass represents one unit of a sleep dataset: a contiguous segment
    of polysomnography (PSG) or electroencephalography (EEG) signal along with its corresponding
    sleep stage annotation from a hypnogram. Epochs are typically 30 seconds in duration and
    are extracted on a uniform grid from the PSG recording, with labels assigned by aligning
    epoch start times to hypnogram segment boundaries.

    Attributes
    ----------
    x : np.ndarray
        Shape (T,), dtype float32. Signal samples for this epoch, representing the raw or
        filtered PSG/EEG measurements over the epoch duration. The number of samples T is
        determined by epoch_duration_s * sampling_rate (e.g., 30s * 256 Hz = 7680 samples).
    label : int
        Integer sleep stage identifier (e.g., 0 for Wake, 1 for N1, 2 for N2, 3 for N3,
        4 for REM). The label maps to a unique label string via the Epochs container's
        unique_labels tuple.

    Notes
    -----
    - The dataclass is frozen (immutable) to ensure epochs cannot be accidentally modified
      after creation, supporting safe use in ML pipelines and data analysis workflows.
    - Epochs are typically not created directly; instead they are generated by
      SleepSession.get_sleep_dataset(), which handles alignment with hypnogram data.
    - The signal array x is stored as float32 for memory efficiency and compatibility
      with deep learning frameworks (PyTorch, TensorFlow).

    Examples
    --------
    >>> import numpy as np
    >>> # Create a single epoch with 7680 samples and sleep stage label (1 = N1)
    >>> signal = np.random.randn(7680).astype(np.float32)
    >>> epoch = Epoch(x=signal, label=1)
    >>> epoch.x.shape
    (7680,)
    >>> epoch.label
    1

    See Also
    --------
    Epochs : Container holding a collection of Epoch objects with shared metadata.
    SleepSession.get_sleep_dataset : Generate epochs from PSG and hypnogram data.
    """

    x: np.ndarray  # shape: (T,), dtype: float32; PSG/EEG signal samples for this epoch
    label: int  # sleep stage label (integer index into unique_labels tuple)


@dataclass(frozen=True)
class Epochs:
    """
    Container for a collection of epochs with shared metadata and statistics.

    This dataclass holds a list of individual Epoch objects (each containing a PSG signal
    segment and its corresponding sleep stage label) along with shared recording parameters,
    temporal information, and dataset statistics. It provides convenient access to the epoch
    collection and methods to convert epochs into stacked numpy arrays suitable for machine
    learning pipelines.

    Attributes
    ----------
    epochs : List[Epoch]
        List of Epoch objects, each containing a signal segment (x) and sleep stage label.
    channel : str
        Name of the PSG/EEG channel from which epochs were extracted (e.g., "EEG Fpz-Cz").
    start_time_abs : float
        Absolute start time of the PSG recording in seconds since epoch (UNIX timestamp).
    epoch_duration_s : float
        Duration of each epoch in seconds (typically 30 for standard sleep scoring).
    sampling_rate : float
        Sampling frequency of the PSG signal in Hertz (Hz).
    unique_labels : tuple[str, ...]
        Tuple of unique sleep stage label names present in the dataset
        (e.g., ("Wake", "N1", "N2", "N3", "REM")).
    alignment_offset_s : float
        Temporal offset in seconds applied during hypnogram-to-epoch alignment.
        Typically 0.0 for synchronous alignment; non-zero if deliberate time shift was applied.
    stats : Dict[str, float]
        Dictionary containing dataset statistics and processing metadata, including:
        - ``total_epochs_grid``: total epochs on the uniform grid before filtering
        - ``kept_epochs``: epochs retained after alignment and validation
        - ``dropped_epochs``: epochs discarded due to invalid labels or boundary issues
        - ``drop_fraction``: fraction of epochs dropped (for validation)
        - ``dropped_no_label``: epochs outside hypnogram coverage
        - ``dropped_nonoverlap``: epochs with sample indices out of bounds
        - ``fs_hz``: sampling frequency (redundant with ``sampling_rate``)
        - ``epoch_len_s``: epoch duration in seconds (redundant with ``epoch_duration_s``)
        - ``epoch_len_samples``: epoch duration in samples
        - ``psg_len_s``: total PSG recording duration in seconds
        - ``hyp_segments``: number of hypnogram segments
        - ``hyp_start_s``: onset time of first hypnogram segment
        - ``hyp_end_s``: end time of last hypnogram segment

    Examples
    --------
    >>> # Convert epochs to stacked arrays for training
    >>> X, y = epochs_container.to_arrays()
    >>> X.shape  # (num_epochs, epoch_samples)
    (150, 7680)
    >>> y.shape  # (num_epochs,)
    (150,)
    """

    epochs: List[Epoch]
    channel: str
    start_time_abs: float  # absolute PSG start time (seconds since epoch)
    epoch_duration_s: float  # e.g., 30
    sampling_rate: float  # Hz
    unique_labels: tuple[str, ...]  # e.g., ("Wake", "N1", "N2", "N3", "REM")
    alignment_offset_s: float  # shift applied to hypnogram times (seconds)
    stats: Dict[str, float]  # drop counts, fractions, epoch/segment counts, etc.

    def __len__(self) -> int:
        """
        Return the number of epochs in this container.

        Returns
        -------
        int
            Total count of Epoch objects in the ``epochs`` list.
        """
        return len(self.epochs)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert epoch collection to stacked numpy arrays for machine learning.

        Stacks all epoch signal segments along a new axis to produce a 2D array
        suitable for training neural networks or other ML models. Labels are
        converted to a 1D integer array with dtype int64.

        Returns
        -------
        X : np.ndarray
            Shape (num_epochs, epoch_samples). Stacked PSG signal segments
            from all Epoch objects, with dtype inherited from individual epoch data
            (typically float32).
        y : np.ndarray
            Shape (num_epochs,). Sleep stage labels (integer indices) for each epoch,
            with dtype int64.

        Raises
        ------
        ValueError
            If no epochs are available (``len(self.epochs) == 0``).

        Examples
        --------
        >>> X, y = epochs_container.to_arrays()
        >>> print(X.shape, y.shape)
        (150, 7680) (150,)
        >>> print(X.dtype, y.dtype)
        float32 int64
        """
        # Validate that at least one epoch is available
        if len(self.epochs) == 0:
            raise ValueError("No epochs available.")

        # Stack signal segments from all epochs along a new axis (axis=0)
        # Result shape: (num_epochs, samples_per_epoch)
        X: np.ndarray = np.stack([e.x for e in self.epochs], axis=0)  # (N, T)

        # Extract and convert labels to int64 array
        # Result shape: (num_epochs,)
        y: np.ndarray = np.asarray(
            [e.label for e in self.epochs], dtype=np.int64
        )  # (N,)

        return X, y


class SleepSession:
    """
    Convenience wrapper combining PSG and hypnogram readers.

    This class provides integrated access to polysomnography (PSG) signals
    and sleep stage annotations (hypnogram). It supports filtering of PSG
    channels and remapping of hypnogram labels, with methods to retrieve
    individual channel data, hypnogram data, or combined plots ready for
    visualization.
    """

    def __init__(self, edf_path: str | Path, hypno_path: str | Path) -> None:
        """
        Create a SleepSession.

        Parameters
        ----------
        edf_path : str | Path
            Path to the PSG/EDF file.
        hypno_path : str | Path
            Path to the hypnogram/annotation file.

        Raises
        ------
        FileNotFoundError
            If either the EDF or hypnogram file does not exist.
        ValueError
            If the EDF or hypnogram file format is invalid.
        """

        # Initialize the PSG reader with the provided EDF file path
        self.psg: PsgReader = PsgReader(edf_path)

        # Initialize the hypnogram reader with the provided annotation file path
        self.hypno: HypnogramReader = HypnogramReader(hypno_path)

    def get_channel_data(
        self,
        *,
        channel: str,
        filter_cfg: FilterConfig,
    ) -> ChannelData:
        """
        Fetch channel data and optionally apply the configured filter.

        Parameters
        ----------
        channel : str
            Name of the channel to retrieve from the PSG (must exist).
        filter_cfg : FilterConfig
            Filter configuration indicating whether to apply band-pass
            filtering and which cutoff frequencies/order to use.

        Returns
        -------
        ChannelData
            Channel data with the same structure returned by ``PsgReader``,
            optionally filtered according to ``filter_cfg``.

        Raises
        ------
        ValueError
            If the specified channel does not exist in the PSG file.
        """

        # Retrieve the raw channel structure from the PSG reader
        channel_data: ChannelData = self.psg.get_channel_data(channel)

        # Apply filtering only when enabled; otherwise pass-through data
        if filter_cfg.enabled:
            # Zero-phase bandpass to avoid phase distortion in EEG/PSG
            y_data: np.ndarray = bandpass_filter(
                signal=channel_data.y_data,
                sampling_frequency=channel_data.sampling_rate,
                l_freq=filter_cfg.l_freq,
                h_freq=filter_cfg.h_freq,
                order=filter_cfg.order,
            )
        else:
            # No filtering: use original signal data
            y_data: np.ndarray = channel_data.y_data

        # Return a ChannelData instance mirroring the original but with
        # the filtered (or passthrough) signal stored in `y_data`.
        return ChannelData(
            x_label=channel_data.x_label,
            x_data=channel_data.x_data,
            y_label=channel_data.y_label,
            y_data=y_data,
            y_scale_factor=channel_data.y_scale_factor,
            sampling_rate=channel_data.sampling_rate,
            start_time=channel_data.start_time,
            recording_duration=channel_data.recording_duration,
        )

    def get_hypnogram_data(
        self,
        *,
        remap: MappingConfig,
    ) -> HypnogramData:
        """
        Retrieve hypnogram data with optional label remapping.

        This method fetches sleep stage annotations from the hypnogram reader
        and optionally remaps the stage labels according to the provided
        mapping configuration. The returned data is structured for direct use
        in plotting and epoch-based analysis.

        Parameters
        ----------
        remap : MappingConfig
            Configuration object controlling label remapping behavior.
            If ``enabled=True``, applies the label transformations specified
            in the ``remap`` dictionary; otherwise returns original labels.

        Returns
        -------
        HypnogramData
            Dataclass containing:
            - ``times``: relative time offsets (seconds) for each epoch
            - ``vals``: integer sleep stage identifiers
            - ``unique_labels``: tuple of stage labels present in the data

        Raises
        ------
        ValueError
            If the remap configuration contains invalid label mappings.

        See Also
        --------
        MappingConfig : Configuration for hypnogram label remapping.
        HypnogramData : Output dataclass structure.
        """
        # Fetch hypnogram annotations from the underlying reader, applying optional
        # label remapping based on the provided MappingConfig. The remap parameter
        # controls whether to transform labels (enabled=True) or use originals (enabled=False).
        hypno_data: HypnogramData = self.hypno.get_hypnogram_data(remap=remap)

        # Return the hypnogram data structure containing times, values, and unique labels
        # ready for downstream visualization and epoch labeling operations
        return hypno_data

    def get_sleep_data_plot(
        self,
        *,
        channel: str,
        filter_cfg: FilterConfig,
        remap_cfg: MappingConfig,
        title: Optional[str] = None,
        restrict_to_hypnogram_window: bool = True,
    ) -> SleepDataPlot:
        """
        Generate combined PSG and hypnogram plot data.

        This method integrates filtered PSG channel data with remapped hypnogram
        annotations and produces a data structure ready for visualization. It handles
        temporal alignment and optional windowing to ensure PSG and hypnogram data
        are synchronized and colocated.

        Parameters
        ----------
        channel : str
            Name of the PSG channel to retrieve (must exist in the EDF file).
        filter_cfg : FilterConfig
            Filter configuration for the PSG channel. If ``enabled=True``,
            applies bandpass filtering; otherwise returns raw signal.
        remap_cfg : MappingConfig
            Configuration for hypnogram label remapping. If ``enabled=True``,
            remaps sleep stage labels; otherwise uses original labels.
        title : Optional[str], optional
            Title for the plot. If None, no title is set. Default is None.
        restrict_to_hypnogram_window : bool, optional
            If True, clips PSG data to the time window covered by the hypnogram.
            If False, includes all PSG data within its recording span.
            Default is True.

        Returns
        -------
        SleepDataPlot
            Dataclass containing synchronized PSG and hypnogram data ready for
            visualization, including x/y data, labels, scales, and tick labels.

        Raises
        ------
        ValueError
            If the specified channel does not exist or remap configuration is invalid.

        See Also
        --------
        get_channel_data : Retrieve and filter individual PSG channels.
        get_hypnogram_data : Retrieve and remap hypnogram annotations.
        SleepDataPlot : Output dataclass structure for plotting.
        """
        # ---- Load PSG channel data ----
        # Retrieve PSG data with optional filtering; x_data contains absolute times (seconds since epoch)
        psg: ChannelData = self.get_channel_data(channel=channel, filter_cfg=filter_cfg)

        # ---- Load hypnogram data ----
        # Retrieve hypnogram annotations with optional label remapping;
        # hypnogram times are in relative seconds from recording start
        hyp: HypnogramData = self.get_hypnogram_data(remap=remap_cfg)

        # ---- Establish temporal alignment ----
        # Extract PSG recording boundaries in absolute time (seconds since epoch)
        psg_start: float = float(psg.start_time)
        psg_end: float = float(psg.start_time + psg.recording_duration)

        # Convert hypnogram times from relative (offset from PSG start) to absolute (seconds since epoch)
        # by adding PSG start time to each hypnogram timestamp
        hyp_time_abs: np.ndarray = psg_start + np.asarray(hyp.times, dtype=float)
        hyp_vals: np.ndarray = np.asarray(hyp.vals, dtype=int)

        # ---- Filter hypnogram to PSG bounds ----
        # Remove hypnogram annotations that fall outside the PSG recording window
        # This prevents plotting data with no corresponding PSG signal
        keep_h: np.ndarray = (hyp_time_abs >= psg_start) & (hyp_time_abs <= psg_end)
        hyp_time_abs = hyp_time_abs[keep_h]
        hyp_vals = hyp_vals[keep_h]

        # ---- Initialize PSG plot data ----
        # Default to the full PSG temporal extent and signal values
        time_data: np.ndarray = psg.x_data
        psg_data: np.ndarray = psg.y_data

        # ---- Optional temporal windowing to hypnogram bounds ----
        # For cleaner visualization, restrict PSG data to the time window covered by hypnogram
        if restrict_to_hypnogram_window and hyp_time_abs.size > 0:
            # Compute hypnogram temporal bounds
            t0: float = float(hyp_time_abs.min())
            t1: float = float(hyp_time_abs.max())

            # Create boolean mask selecting PSG samples within [t0, t1]
            keep_p: np.ndarray = (time_data >= t0) & (time_data <= t1)
            time_data = time_data[keep_p]
            psg_data = psg_data[keep_p]

        # ---- Construct and return plot data container ----
        # Bundle synchronized, filtered, and windowed PSG/hypnogram data for visualization
        return SleepDataPlot(
            psg_x_data=time_data,
            psg_x_label=psg.x_label,
            psg_y_data=psg_data,
            psg_y_label=psg.y_label,
            psg_y_scale=psg.y_scale_factor,
            hypno_x_data=hyp_time_abs,
            hypno_x_label="Time (s)",
            hypno_y_data=hyp_vals,
            hypno_y_label="Sleep Stage",
            hypno_y_tick_labels=hyp.unique_labels,
            title=title,
        )

    def get_sleep_dataset(
        self,
        *,
        channel: str,
        epoch_len_s: int = 30,
        filter_cfg: FilterConfig,
        remap_cfg: MappingConfig,
        max_drop_fraction: float = 0.01,
    ) -> Epochs:
        """
        Build an epoch dataset by aligning PSG/EEG channel data with hypnogram annotations.

        This method creates a uniform grid of fixed-duration epochs from the PSG signal
        and assigns sleep stage labels from the hypnogram by checking which annotated
        segment covers each epoch start time. Epochs are dropped if they fall outside
        hypnogram coverage or if sampling indices exceed signal bounds.

        The hypnogram is provided in step-plot format (alternating onset/offset times with
        repeated labels), which is converted to segments for efficient epoch labeling.

        Parameters
        ----------
        channel : str
            Name of the PSG/EEG channel to extract (must exist in the EDF file).
        epoch_len_s : int, optional
            Duration of each epoch in seconds. Default is 30.
        filter_cfg : FilterConfig
            Filter configuration for the PSG channel. If ``enabled=True``, applies
            bandpass filtering; otherwise returns raw signal.
        remap_cfg : MappingConfig
            Configuration for hypnogram label remapping. If ``enabled=True``, remaps
            sleep stage labels; otherwise uses original labels.
        max_drop_fraction : float, optional
            Maximum allowable fraction of epochs dropped due to missing labels or
            boundary violations. Raises ValueError if exceeded. Default is 0.01 (1%).

        Returns
        -------
        Epochs
            Container holding:
            - ``epochs``: list of Epoch objects with signal samples and labels
            - ``channel``: channel name
            - ``start_time_abs``: absolute PSG start time (seconds since epoch)
            - ``epoch_duration_s``: duration per epoch
            - ``sampling_rate``: PSG sampling rate (Hz)
            - ``unique_labels``: tuple of unique sleep stage labels
            - ``alignment_offset_s``: temporal offset applied (0.0 for this implementation)
            - ``stats``: dictionary with epoch counts, drop statistics, and metadata

        Raises
        ------
        ValueError
            If PSG signal is not 1D, epoch duration is invalid, hypnogram data is
            malformed, no valid epochs remain after alignment, or drop fraction
            exceeds ``max_drop_fraction``.

        See Also
        --------
        Epoch : Individual epoch structure (signal + label).
        Epochs : Container for epoch dataset.
        get_channel_data : Retrieve and optionally filter PSG channel.
        get_hypnogram_data : Retrieve and optionally remap hypnogram annotations.
        """

        # ---- Load PSG/EEG channel ----
        # Retrieve PSG data with optional filtering applied
        psg: ChannelData = self.get_channel_data(channel=channel, filter_cfg=filter_cfg)
        fs: float = float(psg.sampling_rate)  # sampling frequency in Hz
        start_time_abs: float = float(
            psg.start_time
        )  # absolute start time (s since epoch)

        signal: np.ndarray = np.asarray(psg.y_data)  # 1D signal array
        if signal.ndim != 1:
            raise ValueError("Expected a single channel (1D) PSG/EEG signal.")

        # Compute epoch size in samples (e.g., 30s * 256 Hz = 7680 samples)
        epoch_n: int = int(round(float(epoch_len_s) * fs))
        if epoch_n <= 0:
            raise ValueError("epoch_len_s * sampling_rate must be > 0")

        # Total PSG duration derived from sample count (robust against timing quirks)
        psg_len_s: float = float(signal.size) / fs

        # ---- Load hypnogram in step-plot format ----
        # HypnogramReader.get_hypnogram_data() returns times/vals arrays where each
        # annotation contributes [onset, onset+duration] with the label repeated twice
        hyp: HypnogramData = self.get_hypnogram_data(remap=remap_cfg)
        hyp_times: np.ndarray = np.asarray(hyp.times, dtype=float)
        hyp_vals: np.ndarray = np.asarray(hyp.vals, dtype=int)

        if hyp_times.ndim != 1 or hyp_vals.ndim != 1 or hyp_times.size != hyp_vals.size:
            raise ValueError("Hypnogram times/vals must be 1D arrays of equal length.")
        if hyp_times.size < 2:
            raise ValueError("Hypnogram data too short to form segments.")

        # ---- Convert step-plot arrays to segment representation ----
        # Step-plot format: [t0, t1, t0, t1, ...] with labels [l, l, l, l, ...]
        # Extract segments at even indices: onset=times[0], end=times[1], label=vals[0]
        if hyp_times.size % 2 != 0:
            # Defensive measure: drop trailing element if odd-length
            hyp_times = hyp_times[:-1]
            hyp_vals = hyp_vals[:-1]

        seg_onsets: np.ndarray = hyp_times[0::2]  # segment start times
        seg_ends: np.ndarray = hyp_times[1::2]  # segment end times
        seg_labels: np.ndarray = hyp_vals[0::2]  # segment labels

        if seg_onsets.size == 0:
            raise ValueError("No hypnogram segments extracted from step-plot arrays.")

        # ---- Validate and clean segments ----
        # Remove segments with non-positive duration or invalid (infinite/NaN) values
        seg_durs: np.ndarray = seg_ends - seg_onsets
        valid_seg: np.ndarray = (
            (seg_durs > 0) & np.isfinite(seg_onsets) & np.isfinite(seg_ends)
        )
        seg_onsets = seg_onsets[valid_seg]
        seg_ends = seg_ends[valid_seg]
        seg_labels = seg_labels[valid_seg]

        if seg_onsets.size == 0:
            raise ValueError("All hypnogram segments invalid (non-positive duration).")

        # Ensure segments are strictly ordered by onset time (EDF can have quirks)
        order: np.ndarray = np.argsort(seg_onsets)
        seg_onsets = seg_onsets[order]
        seg_ends = seg_ends[order]
        seg_labels = seg_labels[order]

        # ---- Build uniform epoch grid ----
        # Create epoch start times at regular intervals [0, 30, 60, ...]
        epoch_starts: np.ndarray = np.arange(
            0.0, psg_len_s, float(epoch_len_s), dtype=float
        )
        # Keep only epochs that fit completely within PSG duration
        epoch_starts = epoch_starts[epoch_starts + float(epoch_len_s) <= psg_len_s]

        # ---- Assign hypnogram labels to epochs ----
        # For each epoch start, find the segment it belongs to using binary search
        # searchsorted(..., side="right") - 1 gives index of last segment onset <= epoch_start
        idx: np.ndarray = np.searchsorted(seg_onsets, epoch_starts, side="right") - 1

        # Epoch is valid if it falls within its segment: onset <= t < end
        valid_epoch: np.ndarray = (
            idx >= 0
        )  # boolean mask for epochs with a covering segment
        idx2: np.ndarray = idx[valid_epoch]  # segment indices for valid epochs
        starts2: np.ndarray = epoch_starts[valid_epoch]  # starts of valid epochs

        # Check that epoch starts fall before segment end times
        covered: np.ndarray = starts2 < seg_ends[idx2]
        epoch_starts_cov: np.ndarray = starts2[
            covered
        ]  # epochs with valid segment coverage
        epoch_labels_cov: np.ndarray = seg_labels[idx2[covered]].astype(
            int
        )  # corresponding labels

        # Epoch accounting
        total_epochs: int = int(epoch_starts.size)  # total epochs on grid
        dropped_no_label: int = int(
            total_epochs - epoch_starts_cov.size
        )  # epochs outside hypnogram

        # ---- Extract PSG samples for each covered epoch ----
        epochs_list: list[Epoch] = []
        dropped_nonoverlap: int = 0

        for t_rel, lab in zip(epoch_starts_cov, epoch_labels_cov):
            # Convert epoch start time (seconds) to sample index
            s0: int = int(round(float(t_rel) * fs))
            s1: int = s0 + epoch_n  # end sample index
            # Safety check: ensure samples are within signal bounds
            if s0 < 0 or s1 > signal.size:
                dropped_nonoverlap += 1
                continue
            # Extract and convert to float32
            x: np.ndarray = signal[s0:s1].astype(np.float32, copy=False)
            epochs_list.append(Epoch(x=x, label=int(lab)))

        # ---- Final statistics and validation ----
        kept: int = len(epochs_list)
        dropped: int = total_epochs - kept
        drop_fraction: float = (
            dropped / float(total_epochs) if total_epochs > 0 else 1.0
        )

        stats: dict[str, float] = {
            "total_epochs_grid": float(total_epochs),  # all epochs on uniform grid
            "kept_epochs": float(kept),  # epochs with valid label and samples
            "dropped_epochs": float(dropped),  # total dropped
            "drop_fraction": float(
                drop_fraction
            ),  # fraction of total dropped (for validation)
            "dropped_no_label": float(dropped_no_label),  # outside hypnogram coverage
            "dropped_nonoverlap": float(
                dropped_nonoverlap
            ),  # sample index out of bounds (typically near 0)
            "fs_hz": float(fs),  # sampling frequency
            "epoch_len_s": float(epoch_len_s),  # epoch duration in seconds
            "epoch_len_samples": float(epoch_n),  # epoch duration in samples
            "psg_len_s": float(psg_len_s),  # total PSG duration in seconds
            "hyp_segments": float(seg_onsets.size),  # number of hypnogram segments
            "hyp_start_s": float(seg_onsets[0]),  # first segment onset
            "hyp_end_s": float(seg_ends[-1]),  # last segment end
        }

        if kept == 0:
            raise ValueError(f"No epochs left after alignment/rejection. stats={stats}")

        if drop_fraction > float(max_drop_fraction):
            raise ValueError(
                f"Too many epochs dropped ({drop_fraction:.2%} > {max_drop_fraction:.2%}). "
                f"Likely PSG/hypnogram non-overlap or wrong epoching assumptions. stats={stats}"
            )

        return Epochs(
            epochs=epochs_list,
            channel=channel,
            start_time_abs=start_time_abs,
            epoch_duration_s=float(epoch_len_s),
            sampling_rate=float(fs),
            unique_labels=hyp.unique_labels,
            alignment_offset_s=0.0,  # hypnogram times already relative to PSG start
            stats=stats,
        )


def format_sleep_dataset_summary(ds: Epochs) -> str:
    """
    Generate a complete, formatted summary string for a Sleep dataset.

    This function takes an Epochs object and produces a human-readable summary
    including dataset dimensions, sampling parameters, epoch statistics, array
    shapes, and label distribution with percentages.

    Parameters
    ----------
    ds : Epochs
        The sleep dataset epochs container with statistics and label information.

    Returns
    -------
    str
        Formatted multi-line summary string suitable for console output or logging.
    """

    # Basic dimensions: extract total epoch count and samples per epoch
    n_epochs: int = len(ds.epochs)
    epoch_samples: int = ds.epochs[0].x.shape[0] if n_epochs else 0

    # Build stacked arrays once for efficiency (used in output shapes only)
    if n_epochs:
        X: np.ndarray = np.stack([e.x for e in ds.epochs], axis=0)  # (N, T)
        y: np.ndarray = np.asarray([e.label for e in ds.epochs], dtype=np.int64)  # (N,)
    else:
        X = np.empty((0, 0))
        y = np.empty((0,), dtype=np.int64)

    # Core summary block: basic metadata and statistics
    lines: list[str] = [
        "",
        "-" * 40,
        "Sleep Dataset Summary",
        "-" * 40,
        f"Channel              : {ds.channel}",
        f"Sampling Rate (Hz)   : {ds.sampling_rate:.2f}",
        f"Epoch Duration (s)   : {ds.epoch_duration_s:.2f}",
        f"Epoch Samples        : {epoch_samples}",
        f"Total Epochs (Grid)  : {int(ds.stats.get('total_epochs_grid', 0))}",
        f"Kept Epochs          : {int(ds.stats.get('kept_epochs', 0))}",
        f"Dropped Epochs       : {int(ds.stats.get('dropped_epochs', 0))}",
        f"Drop Fraction        : {ds.stats.get('drop_fraction', 0.0):.4f}",
        f"PSG Duration (s)     : {ds.stats.get('psg_len_s', 0.0):.2f}",
        f"Hypnogram Segments   : {int(ds.stats.get('hyp_segments', 0))}",
        f"Unique Labels        : {ds.unique_labels}",
        "",
        "Array Shapes",
        "-" * 40,
        f"X shape : {X.shape}",
        f"y shape : {y.shape}",
        "",
        "Label Distribution",
        "-" * 40,
    ]

    # Label distribution: compute counts and percentages for each unique label
    if y.size > 0:
        unique: np.ndarray
        counts: np.ndarray
        unique, counts = np.unique(
            y, return_counts=True
        )  # sorted unique labels with occurrence counts

        total: float = float(y.size)  # total number of epochs

        # For each unique label, map index to name and format statistics line
        for u, c in zip(unique, counts):
            label_name: str = ds.unique_labels[int(u)]
            lines.append(
                f"{label_name:<20} : {int(c):>5} epochs ({(c/total)*100:6.2f}%)"
            )
    else:
        lines.append("(empty dataset)")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage:
    from . import plotters as plot
    from . import folders as folder

    # --- load a sample sleep session ---
    sample_file_index = 1  # adjust as needed to select a different file
    sample_eeg_file = folder.get_edf_files_in_sleep_data()[sample_file_index]
    sample_hyp_file = folder.get_hypnogram_files_in_sleep_data()[sample_file_index]
    sleep_session = SleepSession(sample_eeg_file, sample_hyp_file)

    # --- original hypnogram plot (no remapping) ---
    hypno_data = sleep_session.get_hypnogram_data(remap=MappingConfig(enabled=False))
    plot.plot_hypnogram_from_annotations(
        times=hypno_data.times,
        vals=hypno_data.vals,
        unique_labels=hypno_data.unique_labels,
        subject=sample_eeg_file.stem,  # use file name as subject identifier
        xlim_s=(20000, 60000),
    )

    # --- mapped hypnogram plot (with remapping) ---
    remap_config = MappingConfig(
        enabled=True,
        remap={
            "Sleep stage W": "Wake",
            "Sleep stage 1": "Sleep",
            "Sleep stage 2": "Sleep",
            "Sleep stage 3": "Sleep",
            "Sleep stage 4": "Sleep",
            "Sleep stage R": "Sleep",
            "Sleep stage ?": "Invalid",
            "Movement time": "Invalid",
        },
    )
    hypno_data_remapped = sleep_session.get_hypnogram_data(remap=remap_config)
    plot.plot_hypnogram_from_annotations(
        times=hypno_data_remapped.times,
        vals=hypno_data_remapped.vals,
        unique_labels=hypno_data_remapped.unique_labels,
        subject=sample_eeg_file.stem,  # use file name as subject identifier
        xlim_s=(20000, 60000),
    )

    # --- raw PSG plot (no filtering) ---
    channel = "EEG Fpz-Cz"  # example channel name; adjust as needed
    eeg_data = sleep_session.get_channel_data(
        channel=channel,
        filter_cfg=FilterConfig(enabled=False),  # no filtering
    )
    plot.plot_signal(
        x_data=eeg_data.x_data,
        y_data=eeg_data.y_data,
        x_label=eeg_data.x_label,
        y_label="uV",  # override with microvolts for EEG
        y_scale=eeg_data.y_scale_factor,
        title=f"Raw PSG Signal for channel {channel}",
        xlim_s=(0, 60),  # show only the first minute for clarity
    )

    # --- filtered PSG plot ---
    channel = "EEG Fpz-Cz"  # example channel name; adjust as needed
    eeg_data = sleep_session.get_channel_data(
        channel=channel,
        filter_cfg=FilterConfig(enabled=True, l_freq=0.5, h_freq=40.0, order=4),
    )
    plot.plot_signal(
        x_data=eeg_data.x_data,
        y_data=eeg_data.y_data,
        x_label=eeg_data.x_label,
        y_label="uV",  # override with microvolts for EEG
        y_scale=eeg_data.y_scale_factor,
        title=f"Filtered PSG Signal for channel {channel}",
        xlim_s=(0, 60),  # show only the first minute for clarity
    )

    # --- combined PSG + hypnogram plot (no filtering, no remapping) ---
    channel = "EEG Fpz-Cz"  # example channel name; adjust as needed
    filter_cfg = FilterConfig(enabled=False)
    remap_cfg = MappingConfig(
        enabled=False,  # use original hypnogram labels for this plot
        remap={},  # no remapping
    )
    sleep_data_plot = sleep_session.get_sleep_data_plot(
        channel=channel,
        filter_cfg=filter_cfg,
        remap_cfg=remap_cfg,
        title=f"Combined PSG and Hypnogram for channel {channel}",
        restrict_to_hypnogram_window=True,  # show only PSG data within hypnogram window
    )
    plot.plot_psg_with_hypnogram_arrays(
        psg_x_data=sleep_data_plot.psg_x_data,
        psg_x_label=sleep_data_plot.psg_x_label,
        psg_y_data=sleep_data_plot.psg_y_data,
        psg_y_label=channel + " (uV)",  # override with microvolts for EEG
        psg_y_scale=sleep_data_plot.psg_y_scale,
        hypno_x_data=sleep_data_plot.hypno_x_data,
        hypno_y_data=sleep_data_plot.hypno_y_data,
        hypno_y_label=sleep_data_plot.hypno_y_label,
        title=sleep_data_plot.title,
        hypno_y_tick=sleep_data_plot.hypno_y_tick_labels,
        xlim_s=(20000, 60000),  # auto-scale x-axis
    )

    # --- combined PSG + hypnogram plot (with filtering and remapping) ---
    channel = "EEG Fpz-Cz"  # example channel name; adjust as needed
    filter_cfg = FilterConfig(enabled=True, l_freq=0.5, h_freq=40.0, order=4)
    remap_cfg = MappingConfig(
        enabled=True,
        remap={
            "Sleep stage W": "Wake",
            "Sleep stage 1": "Sleep",
            "Sleep stage 2": "Sleep",
            "Sleep stage 3": "Sleep",
            "Sleep stage 4": "Sleep",
            "Sleep stage R": "Sleep",
            "Sleep stage ?": "Invalid",
            "Movement time": "Invalid",
        },
    )
    sleep_data_plot = sleep_session.get_sleep_data_plot(
        channel=channel,
        filter_cfg=filter_cfg,
        remap_cfg=remap_cfg,
        title=f"Combined PSG and Hypnogram for channel {channel}",
        restrict_to_hypnogram_window=True,  # show only PSG data within hypnogram window
    )
    plot.plot_psg_with_hypnogram_arrays(
        psg_x_data=sleep_data_plot.psg_x_data,
        psg_x_label=sleep_data_plot.psg_x_label,
        psg_y_data=sleep_data_plot.psg_y_data,
        psg_y_label=channel + " (uV)",  # override with microvolts for EEG
        psg_y_scale=sleep_data_plot.psg_y_scale,
        hypno_x_data=sleep_data_plot.hypno_x_data,
        hypno_y_data=sleep_data_plot.hypno_y_data,
        hypno_y_label=sleep_data_plot.hypno_y_label,
        title=sleep_data_plot.title,
        hypno_y_tick=sleep_data_plot.hypno_y_tick_labels,
        xlim_s=(20000, 60000),  # auto-scale x-axis
    )

    # --- build dataset (single channel + hypnogram) for ML training ---
    ds = sleep_session.get_sleep_dataset(
        channel="EEG Fpz-Cz",
        epoch_len_s=30,
        filter_cfg=FilterConfig(
            enabled=False,  # no filtering for this dataset example
        ),
        remap_cfg=MappingConfig(
            enabled=False,  # no remapping for this dataset example
        ),
    )
    print(format_sleep_dataset_summary(ds))

    # --- build dataset (single channel + hypnogram) for ML training with filter and remapping ---
    ds = sleep_session.get_sleep_dataset(
        channel="EEG Fpz-Cz",
        epoch_len_s=30,
        filter_cfg=FilterConfig(
            enabled=True,  # enable filtering for this dataset example
            l_freq=0.5,
            h_freq=40.0,
            order=4,
        ),
        remap_cfg=MappingConfig(
            enabled=True,  # enable remapping for this dataset example
            remap={
                "Sleep stage W": "Wake",
                "Sleep stage 1": "Sleep",
                "Sleep stage 2": "Sleep",
                "Sleep stage 3": "Sleep",
                "Sleep stage 4": "Sleep",
                "Sleep stage R": "Sleep",
                "Sleep stage ?": "Invalid",
                "Movement time": "Invalid",
            },
        ),
    )
    print(format_sleep_dataset_summary(ds))
