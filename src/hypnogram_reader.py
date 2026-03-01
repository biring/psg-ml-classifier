"""Hypnogram reader utilities.

This module provides a small helper class, :class:`HypnogramReader`, for
loading and working with hypnogram/annotation files commonly used in sleep
research. The class wraps ``mne.Annotations`` to extract stage descriptions,
onset times, and durations and supplies convenient helpers for remapping
stage labels to integer codes and building data suitable for step-style
visualisation (hypnograms).

Key features
------------
- Read annotations from a file using :func:`mne.read_annotations`.
- Expose ordered stage descriptions, onsets, and durations as tuples.
- Provide a stable mapping from stage labels to integers preserving first
    appearance order.
- Build time/value arrays ready for step plotting (start/end points per
    annotation).

Notes
-----
- This module expects annotation files readable by MNE. If a different
    format is required, extend ``HypnogramReader._get_annotation``.
- Methods raise ``RuntimeError`` when required annotation fields are
    missing or malformed.

Example
-------
>>> reader = HypnogramReader('subjectX-hypnogram.edf')
>>> times, vals, labels = reader.get_hypnogram_data()
>>> # then use plotting.plot_hypnogram_from_annotations(times=times, vals=vals, unique_labels=labels)

Dependencies: mne, pathlib
"""

from __future__ import annotations


import mne

from pathlib import Path


class HypnogramReader:
    """
    Reads a hypnogram file and provides utilities to remap and align it to PSG/EEG.

    A hypnogram file contains sleep stage annotations with timing information.
    This class extracts stage labels, onsets, durations, and provides mappings
    for analysis and visualization.

    Default file format:
    - One stage label per line
    - Blank lines ignored
    - Lines starting with '#' ignored

    Attributes:
        hypnogram_file (Path): Path to the hypnogram annotation file.
        annotations (mne.Annotations): MNE annotations object containing timing and labels.
        hypnogram (list[str]): List of stage labels extracted from annotations.
        onsets (tuple[float, ...]): Start times (seconds) for each sleep stage.
        durations (tuple[float, ...]): Duration (seconds) of each sleep stage.
        descriptions (tuple[str, ...]): Stage labels for each annotation.
        unique_stages (tuple[str, ...]): Unique sleep stage labels in order of appearance.
        stage_mapping (dict[str, int]): Mapping from stage labels to integer codes.
    """

    def __init__(self, hypnogram_file: str | Path) -> None:
        """
        Initialize HypnogramReader from a hypnogram file.

        Args:
            hypnogram_file (str | Path): Path to the hypnogram annotation file.

        Raises:
            RuntimeError: If annotations cannot be read from the file.
        """
        self.hypnogram_file: Path = Path(hypnogram_file)
        self.annotations: mne.Annotations = self._get_annotation(self.hypnogram_file)
        self.hypnogram: list[str] = self._extract_hypnogram_stages(self.annotations)
        self.onsets: tuple[float, ...] = self._get_onset()
        self.durations: tuple[float, ...] = self._get_duration()
        self.descriptions: tuple[str, ...] = self._get_description()
        self.unique_stages: tuple[str, ...] = self._get_unique_stages()
        self.stage_mapping: dict[str, int] = self._get_stage_to_int_mapping()

    def _extract_hypnogram_stages(self, annotations: mne.Annotations) -> list[str]:
        """
        Extract hypnogram stages from annotations descriptions.

        Args:
            annotations (mne.Annotations): MNE annotations object.

        Returns:
            list[str]: List of stage labels from annotation descriptions.
        """
        hypnogram: list[str] = []
        if annotations is not None:
            # mne.Annotations.description is already a sequence of strings
            hypnogram = list(annotations.description)
        else:
            # No annotations found; leave hypnogram empty
            hypnogram = []
        return hypnogram

    def _get_annotation(self, hypnogram_file: Path) -> mne.Annotations:
        """
        Read annotations from hypnogram file using MNE.

        Args:
            hypnogram_file (Path): Path to the annotation file.

        Returns:
            mne.Annotations: Annotations object containing timing and labels.

        Raises:
            RuntimeError: If file cannot be read or does not contain valid annotations.
        """
        try:
            return mne.read_annotations(hypnogram_file)
        except Exception:
            raise RuntimeError(
                f"Failed to read annotations from {hypnogram_file} using mne.read_annotations. "
                f"Ensure the file is in a supported format or contains valid annotations."
            )

    def _get_onset(self) -> tuple[float, ...]:
        """
        Extract onset times from annotations.

        Returns:
            tuple[float, ...]: Tuple of onset times in seconds.

        Raises:
            RuntimeError: If annotations are missing, invalid, or contain negative values.
        """
        if getattr(self, "annotations", None) is None:
            raise RuntimeError("No annotations available to extract onsets.")
        onsets = self.annotations.onset
        if onsets is None:
            raise RuntimeError("Annotations do not contain onsets.")
        if onsets.ndim != 1:
            raise RuntimeError(f"Expected 1D onsets array, got shape {onsets.shape}.")
        if (onsets < 0).any():
            raise RuntimeError("Onsets cannot be negative.")
        return tuple(self.annotations.onset)

    def _get_duration(self) -> tuple[float, ...]:
        """
        Extract duration values from annotations.

        Returns:
            tuple[float, ...]: Tuple of duration values in seconds.

        Raises:
            RuntimeError: If annotations are not available.
        """
        if getattr(self, "annotations", None) is None:
            raise RuntimeError("No annotations available to extract durations.")
        return tuple(self.annotations.duration)

    def _get_description(self) -> tuple[str, ...]:
        """
        Extract stage descriptions from annotations.

        Returns:
            tuple[str, ...]: Tuple of stage labels.

        Raises:
            RuntimeError: If annotations are not available.
        """
        if getattr(self, "annotations", None) is None:
            raise RuntimeError("No annotations available to extract descriptions.")
        return tuple(self.annotations.description)

    def _get_unique_stages(self) -> tuple[str, ...]:
        """
        Get unique stage labels in the hypnogram.

        Preserves the order of first appearance while removing duplicates.

        Returns:
            tuple[str, ...]: Tuple of unique stage labels.

        Raises:
            RuntimeError: If no stage descriptions are found.
        """
        # Use dict.fromkeys to preserve insertion order while removing duplicates
        stages: list[str] = list(dict.fromkeys(self.descriptions))
        if len(stages) == 0:
            raise RuntimeError("No stage descriptions found in annotations.")
        return tuple(stages)

    def _get_stage_to_int_mapping(self) -> dict[str, int]:
        """
        Create a mapping from stage labels to integer codes.

        Args:
            None

        Returns:
            dict[str, int]: Dictionary mapping stage labels to sequential integers.
        """
        return {stage: i for i, stage in enumerate(self.unique_stages)}

    def get_hypnogram_data(
        self,
    ) -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...]]:
        """
        Build step-plot data from hypnogram annotations.

        Creates x-coordinates (times), y-coordinates (stage codes), and labels
        suitable for step-plot visualization.

        Returns:
            tuple: A 3-tuple containing:
                - times (tuple[float, ...]): X-coordinates (seconds) for step plot.
                - vals (tuple[int, ...]): Y-coordinates (integer stage codes).
                - unique_labels (tuple[str, ...]): Unique stage labels for legend.
        """
        times: list[float] = []
        vals: list[int] = []
        unique_labels: list[str] = list(self.unique_stages)
        label_to_int: dict[str, int] = self.stage_mapping

        # For each annotation, add two points to create step-plot effect
        for o, d, s in zip(self.onsets, self.durations, self.descriptions):
            # Add start and end time with constant value
            times.extend([o, o + d])
            vals.extend([label_to_int[s], label_to_int[s]])

        return tuple(times), tuple(vals), tuple(unique_labels)


if __name__ == "__main__":
    # Example usage:
    from . import plotters as plot
    from . import folders as folder

    # Read annotations
    sample_hyp_file = folder.get_hypnogram_files_in_sleep_data()[0]  # any subject
    hyp_data = HypnogramReader(sample_hyp_file)

    # Instance variables
    print("Onsets:", hyp_data.onsets)
    print("Durations:", hyp_data.durations)
    print("Descriptions:", hyp_data.descriptions)
    print("Unique stages:", hyp_data.unique_stages)
    print("Stage to int mapping:", hyp_data.stage_mapping)

    # Plot hypnogram
    times, vals, unique_labels = hyp_data.get_hypnogram_data()
    print("Times for plotting:", times)
    print("Values for plotting:", vals)
    print("Unique labels for plotting:", unique_labels)
    plot.plot_hypnogram_from_annotations(
        times=times,
        vals=vals,
        unique_labels=unique_labels,
        subject="Example Subject",
        xlim_s=(25000, 50000),
    )
