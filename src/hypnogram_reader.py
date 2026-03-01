"""Hypnogram reader utilities.

Utilities for loading and converting hypnogram/annotation files used in
sleep research. The primary class, :class:`HypnogramReader`, reads
MNE-compatible annotations and produces plotting-friendly, step-style
hypnogram data via the :class:`HypnogramData` dataclass.

Key features
------------
- Read annotations using ``mne.read_annotations``.
- Extract descriptions, onsets, and durations from annotations.
- Preserve first-appearance order when mapping labels to integers.
- Optionally remap labels (e.g., combine stages) via ``MapppingConfig``.

Public API
----------
- ``HypnogramData``: dataclass with ``times``, ``vals``, and
  ``unique_labels`` for plotting.
- ``HypnogramReader``: loads annotations and converts them to
  ``HypnogramData``.

Notes
-----
- This module expects files readable by MNE. To support other formats,
  override ``HypnogramReader._get_annotation``.
- Methods raise ``RuntimeError`` for missing or malformed annotation data.

Example
-------
>>> reader = HypnogramReader('subj-hypnogram.edf')
>>> data = reader.get_hypnogram_data(remap=MapppingConfig(enabled=False, remap={}))
"""

from __future__ import annotations
from dataclasses import dataclass, field


import mne

from pathlib import Path


@dataclass(frozen=True)
class HypnogramData:
    times: tuple[float, ...]
    vals: tuple[int, ...]
    unique_labels: tuple[str, ...]


@dataclass(frozen=True)
class MapppingConfig:
    enabled: bool = False
    remap: dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"N1": "Light sleep", "N2": "Light sleep", ...}


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

    def _apply_remap_and_validate(
        self, remap: MapppingConfig | None
    ) -> tuple[list[str], tuple[str, ...], dict[str, int]]:
        """
        Validate a remapping configuration and produce remapped descriptions,
        the unique mapped labels (preserving order), and a new label->int map.

        Validation rules:
        - If ``remap`` is None or ``remap.enabled`` is False, the original
          descriptions and mapping are returned unchanged.
        - The remap dictionary must contain exactly the set of unique raw
          stage labels (no missing or unknown keys).
        - Mapped values must be non-empty strings.

        Raises ValueError on invalid remap config.
        """
        raw_unique = set(self.unique_stages)
        remap_keys = set(remap.remap.keys())

        missing = raw_unique - remap_keys
        extra = remap_keys - raw_unique
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing mappings for raw labels: {sorted(missing)}")
            if extra:
                parts.append(f"unknown remap keys: {sorted(extra)}")
            raise ValueError("Invalid remap config: " + "; ".join(parts))

        # Apply remap to each description in original order
        mapped_descriptions: list[str] = [remap.remap[d] for d in self.descriptions]

        # Ensure mapped values are non-empty strings
        if any((not isinstance(v, str) or v == "") for v in mapped_descriptions):
            raise ValueError("Mapped labels must be non-empty strings")

        # Preserve first-appearance order for the new unique labels
        mapped_unique: tuple[str, ...] = tuple(dict.fromkeys(mapped_descriptions))

        new_mapping: dict[str, int] = {lab: i for i, lab in enumerate(mapped_unique)}

        return mapped_descriptions, mapped_unique, new_mapping

    def get_hypnogram_data(
        self,
        *,
        remap: MapppingConfig,
    ) -> HypnogramData:
        """
        Build step-plot data from hypnogram annotations.

        Creates x-coordinates (times), y-coordinates (stage codes), and labels
        suitable for step-plot visualization.

        Returns:
            HypnogramData: Data class containing times, stage codes, and unique labels for step-plot visualization.
        """
        if not isinstance(remap, MapppingConfig):
            raise TypeError("remap must be an instance of MapppingConfig")

        if remap.enabled:
            # Apply remapping to hypno labels (validates remap covers raw labels)
            mapped_descriptions, mapped_unique, label_to_int = (
                self._apply_remap_and_validate(remap)
            )
        else:
            mapped_descriptions = tuple(self.descriptions)
            mapped_unique = tuple(self.unique_stages)
            label_to_int = self.stage_mapping

        times: list[float] = []
        vals: list[int] = []

        # For each annotation, add two points to create step-plot effect
        for o, d, s in zip(self.onsets, self.durations, mapped_descriptions):
            # Add start and end time with constant value
            times.extend([o, o + d])
            vals.extend([label_to_int[s], label_to_int[s]])

        return HypnogramData(
            times=tuple(times), vals=tuple(vals), unique_labels=tuple(mapped_unique)
        )


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

    # Plot default hypnogram
    map_config = MapppingConfig(
        enabled=False,
        remap={},  # no remapping; use original labels},
    )
    hypnogram_data = hyp_data.get_hypnogram_data(
        remap=map_config,
    )
    print("Times for plotting:", hypnogram_data.times)
    print("Values for plotting:", hypnogram_data.vals)
    print("Unique labels for plotting:", hypnogram_data.unique_labels)
    plot.plot_hypnogram_from_annotations(
        times=hypnogram_data.times,
        vals=hypnogram_data.vals,
        unique_labels=hypnogram_data.unique_labels,
        subject="Example Subject",
        xlim_s=(25000, 50000),
    )

    # Plot remapped hypnogram (e.g., combine N1 and N2 into "Light sleep")
    remap_config = MapppingConfig(
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
    hypnogram_data = hyp_data.get_hypnogram_data(
        remap=remap_config,
    )
    print("Times for plotting:", hypnogram_data.times)
    print("Values for plotting:", hypnogram_data.vals)
    print("Unique labels for plotting:", hypnogram_data.unique_labels)
    plot.plot_hypnogram_from_annotations(
        times=hypnogram_data.times,
        vals=hypnogram_data.vals,
        unique_labels=hypnogram_data.unique_labels,
        subject="Example Subject",
        xlim_s=(25000, 50000),
    )
