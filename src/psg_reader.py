"""PSG (Polysomnography) reader utilities.

This module contains :class:`PsgReader`, a thin wrapper around MNE-Python
utilities for reading EDF-formatted PSG recordings and exposing commonly
used metadata and per-channel signal access in a project-friendly form.

Features
--------
- Read EDF files using ``mne.io.read_raw_edf`` with controlled warning
    suppression for common header inconsistencies.
- Expose EDF header fields (units, filters, samples per record, number of
    records) and derived values such as per-channel sampling rates and total
    sample counts.
- Provide a convenience method, :meth:`get_channel_data`, that returns a
    time axis, the channel signal, human-readable labels, and a scaling
    factor suitable for plotting.

Behavior notes
--------------
- Methods raise ``ValueError`` when required EDF header fields are missing
    or malformed; callers should catch these for robust pipelines.
- The reader intentionally does not preload signal data (``preload=False``)
    to avoid large memory usage; :meth:`get_channel_data` retrieves channel
    data on demand via the underlying MNE ``Raw`` object.

Dependencies: mne, numpy, pathlib
"""

from __future__ import annotations

import warnings
import mne
import numpy as np
from pathlib import Path


class PsgReader:
    """
    Class to read and parse PSG (Polysomnography) EDF files.

    This class reads EDF (European Data Format) files using MNE-Python and extracts
    polysomnography metadata including channel information, sampling rates, filters,
    and signal data.

    Attributes:
        edf_file_path (Path): Path to the EDF file.
        raw (mne.io.BaseRaw): MNE raw object containing the EDF data.
        extras (dict): EDF header metadata from MNE raw extras.
        no_of_data_blocks (int): Number of records/blocks in the EDF file.
        no_of_channels (int): Total number of channels in the recording.
        duration_of_data_block (float): Duration of each record in seconds.
        channel_names (tuple[str]): Names of all channels.
        channel_scale (tuple[float]): Scale/unit conversion factors for each channel.
        channel_lpf (tuple[float]): Low-pass filter frequencies (Hz) for each channel.
        channel_hpf (tuple[float]): High-pass filter frequencies (Hz) for each channel.
        ch_sampling_rate (tuple[float]): Sampling rate (Hz) for each channel.
        ch_sample_count (tuple[int]): Total sample count for each channel.
    """

    def __init__(self, edf_file_path: Path) -> None:
        """
        Initialize PsgReader by reading and parsing an EDF file.

        Args:
            edf_file_path (Path): Path to the EDF file to read.

        Raises:
            ValueError: If the EDF file cannot be read or required metadata is missing.
        """
        self.edf_file_path: Path = edf_file_path
        self.raw: mne.io.BaseRaw = self._read_edf_file()
        self.extras: dict = self._get_extras()
        self.no_of_data_blocks: int = self._get_no_of_data_blocks()
        self.no_of_channels: int = self._get_no_of_channels()
        self.duration_of_data_block: float = self._get_duration_of_data_block()
        self.channel_names: tuple[str, ...] = self._get_channel_names()
        self.channel_scale: tuple[float, ...] = self._get_channel_scale()
        self.channel_lpf: tuple[float, ...] = self._get_channel_low_pass_filter()
        self.channel_hpf: tuple[float, ...] = self._get_channel_high_pass_filter()
        self.ch_sampling_rate: tuple[float, ...] = self._get_sampling_rate_by_channel()
        self.ch_sample_count: tuple[int, ...] = self._get_sample_count_per_channel()

    def _read_edf_file(self) -> mne.io.BaseRaw:
        """
        Read the EDF file using MNE-Python, suppressing known filter-related warnings.

        Returns:
            mne.io.BaseRaw: MNE raw object containing the EDF data.

        Raises:
            ValueError: If the EDF file cannot be read.
        """
        try:
            # Suppress expected warnings about mismatched filter specifications
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Channels contain different highpass filters.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Channels contain different lowpass filters.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Highpass cutoff frequency .* is greater than lowpass cutoff frequency .*",
                    category=RuntimeWarning,
                )
                # Read the EDF file without preloading data into memory
                self.raw = mne.io.read_raw_edf(
                    str(self.edf_file_path), preload=False, verbose=False
                )
                return self.raw
        except Exception as e:
            raise ValueError(f"Failed to read EDF file: {self.edf_file_path}") from e

    def _get_extras(self) -> dict:
        """
        Extract EDF header metadata from the MNE raw object.

        Returns:
            dict: Dictionary containing EDF header extra information.

        Raises:
            ValueError: If extras are not available or not in the expected dict format.
        """
        try:
            # Access the first (and typically only) extras dict from raw object
            self.extras = self.raw._raw_extras[0]

        except Exception as e:
            raise ValueError(
                "EDF header extras not available in MNE raw object."
            ) from e

        if not isinstance(self.extras, dict):
            raise ValueError("EDF header extras not available; expected a dict.")

        return self.extras

    # --- File level metadata ---

    def _get_no_of_channels(self) -> int:
        """
        Extract the total number of channels from the EDF file.

        Returns:
            int: Number of channels in the recording.

        Raises:
            ValueError: If channel count is missing or invalid.
        """
        # Get channel count from MNE info dictionary
        no_of_channels: int | None = self.raw.info.get("nchan", None)

        if no_of_channels is None:
            raise ValueError("Missing number of channels in MNE raw info.")

        no_of_channels = int(no_of_channels)

        # Validate that channel count is positive
        if no_of_channels <= 0:
            raise ValueError(f"Invalid number of channels: {no_of_channels}")

        return no_of_channels

    def _get_no_of_data_blocks(self) -> int:
        """
        Get the number of data blocks (records) from the EDF file using MNE raw extras.

        In EDF format, a "record" is a fixed-duration block of data.

        Returns:
            int: Number of records in the EDF file.

        Raises:
            ValueError: If record count is missing or invalid.
        """
        # Extract number of records from EDF header metadata
        no_records: int | None = self.extras.get("n_records", None)

        if no_records is None:
            raise ValueError("Missing number of records in MNE raw object.")

        no_records = int(no_records)

        # Ensure record count is positive
        if no_records <= 0:
            raise ValueError(f"Invalid number of records: {no_records}")

        return no_records

    # --- Channel level metadata ---

    def _get_channel_names(self) -> tuple[str, ...]:
        """
        Extract channel names from the EDF file.

        Returns:
            tuple[str, ...]: Tuple of channel name strings.

        Raises:
            ValueError: If no channel names are found.
        """
        # Get channel names from MNE raw object
        channel_names: list[str] = self.raw.ch_names

        if not channel_names:
            raise ValueError(
                f"No channel names found in EDF file: {self.edf_file_path}"
            )

        return tuple(channel_names)

    def _get_channel_scale(self) -> tuple[float, ...]:
        """
        Extract channel scale factors (units) from the EDF file.

        The scale factor represents the physical unit for each channel
        (e.g., µV for voltage, mV for millivolts).

        Returns:
            tuple[float, ...]: Scale factors for each channel.

        Raises:
            ValueError: If scale information is missing, mismatched, or invalid.
        """
        # Get unit/scale information from EDF header
        ch_scale: list[str] | None = self.extras.get("units", None)

        if ch_scale is None:
            raise ValueError(
                "Missing channel scale (unit) information in MNE extra object."
            )

        # Ensure scale list matches channel count
        if len(ch_scale) != self.no_of_channels:
            raise ValueError(
                "Mismatch between number of channel units and channel names."
            )

        # Convert scale strings to floats
        scales: list[float] = []
        for s in ch_scale:
            try:
                scale: float = float(s)
                scales.append(scale)
            except ValueError:
                raise ValueError(
                    f"Invalid channel unit value: {s}. Expected a float, got {type(s)}."
                )

        return tuple(scales)

    def _get_channel_low_pass_filter(self) -> tuple[float, ...]:
        """
        Extract low-pass filter frequencies for each channel.

        Low-pass filters attenuate frequencies above the cutoff frequency.
        Missing values are set to 999 Hz (effectively no filtering).

        Returns:
            tuple[float, ...]: Low-pass filter frequencies (Hz) for each channel.

        Raises:
            ValueError: If filter information is missing or invalid.
        """
        # Get low-pass filter cutoff frequencies from EDF header
        ch_lpf: list[str] | None = self.extras.get("lowpass", None)

        if ch_lpf is None:
            raise ValueError(
                "Missing channel lowpass filter information in MNE extra object."
            )

        # Ensure filter list matches channel count
        if len(ch_lpf) != self.no_of_channels:
            raise ValueError(
                "Mismatch between number of channel lowpass filters and channel names."
            )

        # Convert filter strings to floats, handling empty values
        filters: list[float] = []
        for lpf in ch_lpf:
            # Remove whitespace and handle empty strings
            s: str = str(lpf).strip()
            if s == "":
                # No low pass filter specified, set to a very high value as default
                s = "999"

            try:
                filters.append(float(s))
            except ValueError:
                raise ValueError(f"Invalid low pass filter value: '{s}'")

        return tuple(filters)

    def _get_channel_high_pass_filter(self) -> tuple[float, ...]:
        """
        Extract high-pass filter frequencies for each channel.

        High-pass filters attenuate frequencies below the cutoff frequency.
        Missing values are set to 0.001 Hz (effectively no filtering).

        Returns:
            tuple[float, ...]: High-pass filter frequencies (Hz) for each channel.

        Raises:
            ValueError: If filter information is missing or invalid.
        """
        # Get high-pass filter cutoff frequencies from EDF header
        ch_hpf: list[str] | None = self.extras.get("highpass", None)

        if ch_hpf is None:
            raise ValueError(
                "Missing channel highpass filter information in MNE raw object."
            )

        # Ensure filter list matches channel count
        if len(ch_hpf) != self.no_of_channels:
            raise ValueError(
                "Mismatch between number of channel highpass filters and channel names."
            )

        # Convert filter strings to floats, handling empty values
        filters: list[float] = []
        for hpf in ch_hpf:
            # Remove whitespace and handle empty strings
            s: str = str(hpf).strip()
            if s == "":
                # No high pass filter specified, set to very low value
                s = "0.001"
            try:
                filters.append(float(s))
            except ValueError:
                raise ValueError(f"Invalid high pass filter value: '{s}'")

        return tuple(filters)

    # --- Derived data ---

    def _get_duration_of_data_block(self) -> float:
        """
        Calculate the duration of each data block (record) in seconds.

        Uses the EDF definition: record_duration = (nsamples / sfreq) / n_records

        Returns:
            float: Duration of each record in seconds.

        Raises:
            ValueError: If required metadata is missing or invalid.
        """
        # Get total samples across all channels from EDF header
        nsamples: str | None = self.extras.get("nsamples", None)
        if nsamples is None:
            raise ValueError("Missing 'nsamples' in EDF header metadata.")
        try:
            nsamples_float: float = float(nsamples)
        except ValueError:
            raise ValueError(
                f"Invalid 'nsamples' value: {nsamples}. Expected a float got {type(nsamples)}."
            )
        if nsamples_float <= 0:
            raise ValueError(
                f"Invalid 'nsamples' value: {nsamples_float}. Must be positive."
            )

        # Get number of records from EDF header
        n_records: int | None = self.extras.get("n_records", None)
        if n_records is None:
            raise ValueError("Missing 'n_records' in EDF header metadata.")
        try:
            n_records_float: float = float(n_records)
        except ValueError:
            raise ValueError(
                f"Invalid 'n_records' value: {n_records}. Expected a float got {type(n_records)}."
            )
        if n_records_float <= 0:
            raise ValueError(
                f"Invalid 'n_records' value: {n_records_float}. Must be positive."
            )

        # Get sampling frequency from MNE raw info
        sfreq: float | None = self.raw.info.get("sfreq", None)
        if sfreq is None:
            raise ValueError("Missing 'sfreq' in MNE raw.info.")
        try:
            sfreq_float: float = float(sfreq)
        except ValueError:
            raise ValueError(
                f"Invalid 'sfreq' value: {sfreq}. Expected a float got {type(sfreq)}."
            )
        if sfreq_float <= 0:
            raise ValueError(f"Invalid 'sfreq' value: {sfreq_float}. Must be positive.")

        # Calculate record duration
        duration: float = (nsamples_float / sfreq_float) / n_records_float
        if duration <= 0:
            raise ValueError(
                f"Calculated record duration is invalid: {duration} seconds. Check 'nsamples', 'sfreq', and 'n_records' values."
            )

        return duration

    def _get_sample_count_per_channel(self) -> tuple[int, ...]:
        """
        Calculate total sample count per channel across the entire recording.

        Returns:
            tuple[int, ...]: Total number of samples for each channel.

        Raises:
            ValueError: If required metadata is missing or mismatched.
        """
        # Get number of records from EDF header
        n_records: int | None = self.extras.get("n_records", None)
        if n_records is None:
            raise ValueError("Missing 'n_records' in EDF header metadata.")
        try:
            n_records_int: int = int(n_records)
        except ValueError:
            raise ValueError(
                f"Invalid 'n_records' value: {n_records}. Expected an integer got {type(n_records)}."
            )

        # Get samples per record for each channel from EDF header
        n_samps: list[int] | None = self.extras.get("n_samps", None)
        if n_samps is None:
            raise ValueError("Missing 'n_samps' in EDF header metadata.")
        if len(n_samps) != self.no_of_channels:
            raise ValueError("Mismatch between number of channels and n_samps length.")

        # Calculate total samples: n_records * n_samps[i] for each channel
        samples: list[int] = []
        for sample in n_samps:
            try:
                # Multiply samples per record by number of records
                total: int = int(n_records_int * int(sample))
                samples.append(total)
            except ValueError:
                raise ValueError(
                    f"Invalid 'n_samps' value: {sample}. Expected an integer got {type(sample)}."
                )

        return tuple(samples)

    def _get_sampling_rate_by_channel(self) -> tuple[float, ...]:
        """
        Calculate per-channel sampling rates using the EDF definition.

        Sampling rate = n_samps / record_duration (Hz)

        Returns:
            tuple[float, ...]: Sampling rate (Hz) for each channel.

        Raises:
            ValueError: If required metadata is missing or calculated rate is invalid.
        """
        # Get samples per record for each channel from EDF header
        samples: list[int] | None = self.extras.get("n_samps", None)
        if samples is None:
            raise ValueError("Missing 'n_samps' in EDF header metadata.")
        if len(samples) != self.no_of_channels:
            raise ValueError("Mismatch between channel count and n_samps length.")

        # Calculate sampling rate for each channel
        out: list[float] = []
        for sample in samples:
            try:
                sample_float: float = float(sample)
            except ValueError:
                raise ValueError(
                    f"Invalid 'n_samps' value: {sample}. Expected a float got {type(sample)}."
                )
            # Divide samples per record by record duration to get Hz
            sampling_rate: float = sample_float / self.duration_of_data_block
            out.append(sampling_rate)

            # Validate calculated sampling rate is positive
            if out[-1] <= 0:
                raise ValueError(
                    f"Calculated sampling rate is invalid: {out[-1]} Hz. Check 'n_samps' and record duration values."
                )

        return tuple(out)

    def get_channel_data(
        self, channel: str
    ) -> tuple[str, np.ndarray, str, np.ndarray, float]:
        """
        Retrieve time and signal data for a specific channel.

        Returns the time axis, signal data, metadata labels, and scaling factor.

        Args:
            channel (str): Name of the channel to retrieve data for.

        Returns:
            tuple[str, np.ndarray, str, np.ndarray, float]: A tuple containing:
                - x_label (str): Label for time axis with units
                - x_data (np.ndarray): Time values in seconds
                - y_label (str): Label for signal axis with units
                - y_data (np.ndarray): Signal values
                - y_scale_factor (float): Scale factor for the signal

        Raises:
            ValueError: If the channel name is not found or index is out of range.
        """
        # Validate channel name exists
        if channel not in self.channel_names:
            raise ValueError("Channel name not found.")

        # Get the index of the requested channel
        channel_index: int = self.channel_names.index(channel)

        # Validate channel index is within valid range
        if channel_index < 0 or channel_index >= self.no_of_channels:
            raise ValueError("Channel index out of range.")

        # Get sampling frequency for this channel
        freq: float = self.ch_sampling_rate[channel_index]

        # ---- Time axis (x) ----
        x_name: str = "Time"
        x_unit: str = "seconds"
        x_label: str = f"{x_name} ({x_unit})"

        # Create time axis from 0 to duration based on sample count
        n_samples: int = self.ch_sample_count[channel_index]
        x_data: np.ndarray = np.arange(n_samples) / freq

        # ---- Signal axis (y) ----
        y_name: str = self.channel_names[channel_index]
        y_unit: float = self.channel_scale[channel_index]
        y_label: str = f"{y_name} ({y_unit})"

        # Extract signal data for the specified channel
        y_data: np.ndarray = self.raw.get_data(picks=[channel])[0]

        # Calculate scale factor (inverse of unit to convert to base units)
        y_scale: float = self.channel_scale[channel_index]
        y_scale_factor: float = 1 / y_scale if y_scale != 0 else 1.0

        return x_label, x_data, y_label, y_data, y_scale_factor


if __name__ == "__main__":
    # Example usage
    from . import folders as folder
    from . import plotters as plot

    # Read EDF file
    sample_edf_file = folder.get_edf_files_in_sleep_data()[0]  # any subject
    edf_data = PsgReader(sample_edf_file)

    # Instance variables
    print("Number of channels:", edf_data.no_of_channels)
    print("Number of data blocks:", edf_data.no_of_data_blocks)
    print("Duration per data block (seconds):", edf_data.duration_of_data_block)
    print("Channels names:", edf_data.channel_names)
    print("Channel scale:", edf_data.channel_scale)
    print("Channel low pass filter (Hz):", edf_data.channel_lpf)
    print("Channel high pass filter (Hz):", edf_data.channel_hpf)
    print("Channel sample count:", edf_data.ch_sample_count)
    print("Channel sampling rate (Hz):", edf_data.ch_sampling_rate)

    # Plot one channel's data
    channel: str = "EEG Fpz-Cz"  # any channel index
    x_label, x_data, y_label, y_data, y_scale = edf_data.get_channel_data(channel)
    plot.plot_signal(
        x_data=x_data,
        y_data=y_data,
        x_label=x_label,
        y_label="µV",
        y_scale=y_scale,
        title=f"PSG Signal for channel {channel}",
        xlim_s=(0, 60),  # show only the first minute for clarity
    )
