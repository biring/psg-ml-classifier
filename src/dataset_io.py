"""
Utilities to save and load feature `Dataset` objects.

This module provides a lightweight API for Dataset serialization and deserialization
using Python's `pickle` module. It includes functions for saving datasets to disk,
loading them back, and comparing Dataset instances for equality.

Key functions:
- save_dataset_to_file: Save a Dataset to disk in the project's standard format
- load_dataset_from_file: Load a Dataset from disk
- load_all_datasets: Load all datasets from the project's datasets directory
- _save_dataset: Low-level pickle serialization with directory creation
- _load_dataset: Low-level pickle deserialization
- _equals: Deep comparison of two Dataset instances
- _build_file_path: Construct standardized file paths for datasets

The module depends on the `Dataset` dataclass and related utilities defined
in `constants` and `folders` modules.
"""

from __future__ import annotations

# library imports
import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# local imports
from . import constants as const
from . import folders as folder


@dataclass(frozen=True)
class Dataset:
    """
    Immutable container for feature-extracted EEG dataset with labeled epochs.

    This class holds processed EEG epochs where each epoch contains both time-domain
    and frequency-domain features, along with corresponding labels and extraction metadata.
    The frozen dataclass ensures immutability after instantiation.

    Attributes
    ----------
    subject_id : str
        Unique identifier for the subject associated with this dataset (e.g., "SUBJ_001").
    epochs : tuple[tuple[np.ndarray, ...], int]
        Processed epoch represented as a tuple with two elements:
        - [0]: tuple of feature arrays (e.g., (time_domain_features, frequency_domain_features))
        - [1]: integer label index corresponding to a label in y_labels.
    y_labels : tuple[str, ...]
        Tuple of label names (e.g., ("Wake", "N1", "N2", "N3", "REM")).
        Index position in this tuple corresponds to label integer values in epochs.
    stats : dict[str, float]
        Dictionary of metadata captured during feature extraction, such as sampling
        frequency, Nyquist frequency, number of bins, and feature dimensions.
    """

    subject_id: str
    epochs: tuple[tuple[np.ndarray, ...], int]
    y_labels: tuple[str, ...]
    stats: dict[str, float]


def _build_file_path(subject_id: str, tags: tuple[str, ...]) -> Path:
    """
    Build the file path for saving a Dataset.

    Constructs a file path for storing a Dataset pickle file in the project's
    datasets directory, using the subject ID and optional tags to create a
    unique filename.

    Parameters
    ----------
    subject_id : str
        Unique identifier for the subject.
    tags : tuple[str, ...]
        Optional tuple of string tags to include in the filename.
        If empty, only subject_id and file extension are used.

    Returns
    -------
    Path
        The full file path where the Dataset should be saved.
    """
    # Get the project's designated datasets directory
    dataset_folder: Path = folder.get_datasets_dir()

    # Construct filename: subject_id + optional tags + file extension.
    # Note: When `tags` is empty, `"_".join(tags)` yields an empty string,
    # which may produce a double underscore in the filename. We preserve
    # the existing naming convention rather than changing filename layout.
    file_name_str: str = (
        subject_id + "_" + "_".join(tags) + "_" + const.PICKLE_FILE_EXTENSION
    )

    # Return the full path combining directory and filename
    return dataset_folder / f"{file_name_str}"


def _equals(dataset_a: Dataset, dataset_b: Dataset) -> bool:
    """
    Compare two Dataset instances for equality.

    Performs a deep comparison of all Dataset attributes including epochs,
    labels, subject ID, and statistics. Uses numpy-safe array comparison
    to handle numpy arrays in the epochs field.

    Parameters
    ----------
    dataset_a : Dataset
        The first Dataset instance to compare.
    dataset_b : Dataset
        The second Dataset instance to compare.

    Returns
    -------
    bool
        True if both datasets are equal in all attributes, False otherwise.
    """
    # Type check: ensure both arguments are Dataset instances
    if not isinstance(dataset_a, Dataset) or not isinstance(dataset_b, Dataset):
        return False

    # Compare label names
    if dataset_a.y_labels != dataset_b.y_labels:
        return False

    # Compare subject identifiers
    if dataset_a.subject_id != dataset_b.subject_id:
        return False

    # Compare extraction metadata
    if dataset_a.stats != dataset_b.stats:
        return False

    # Compare epoch label indices (the second element in `epochs` tuples)
    if dataset_a.epochs[1] != dataset_b.epochs[1]:
        return False

    # Extract feature arrays from both datasets.
    # `epochs[0]` is expected to be a tuple of numpy arrays holding features.
    arrays_a: tuple[np.ndarray, ...] = dataset_a.epochs[0]
    arrays_b: tuple[np.ndarray, ...] = dataset_b.epochs[0]

    # Check that both datasets have the same number of feature arrays
    if len(arrays_a) != len(arrays_b):
        return False

    # Compare each feature array element-wise using numpy's safe comparison
    for a, b in zip(arrays_a, arrays_b):
        if not np.array_equal(a, b):
            return False

    return True


def _save_dataset(path: str | Path, dataset: Dataset) -> None:
    """
    Save a Dataset instance to disk using pickle serialization.
    This function creates parent directories as needed and supports optional
    gzip compression for reduced file size.

    Parameters
    ----------
    path : str | Path
        Destination file path. Parent directories will be created if needed.
    dataset : Dataset
        The Dataset instance to be serialized and saved.
    Returns
    -------
    None
    Raises
    ------
    pickle.PicklingError
        If the dataset cannot be serialized by pickle.
    OSError
        If the file cannot be written to the specified path.
    Examples
    --------
    >>> _save_dataset('data/my_dataset.pkl', my_dataset)
    >>> _save_dataset('data/my_dataset.pkl.gz', my_dataset, compress=True)

    """
    try:
        # Convert path to Path object for cross-platform compatibility
        file_path: Path = Path(path)

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # if file exists, ignore
        if file_path.exists():
            print("e", end="")  # Print "e" to indicate existing file (without newline)
            return

        # Pre-validate that the dataset can be pickled before writing to disk.
        # This allows catching a `pickle.PicklingError` early, before creating files.
        pickle.dumps(dataset)

        # Write uncompressed pickle directly to file.
        with open(file_path, "wb") as fh:
            pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # check file creation
        if file_path.exists():
            print("c", end="")  # Print "c" to indicate created file (without newline)
            return

    except pickle.PicklingError as e:
        raise pickle.PicklingError(f"Failed to serialize dataset: {e}")
    except OSError as e:
        raise OSError(f"Failed to write dataset to {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during dataset save: {e}")


def _load_dataset(path: str | Path) -> Dataset:
    """
    Load and return a `Dataset` previously saved with :func:`_save_dataset`.

    Deserializes a Dataset instance from disk using pickle.

    Parameters
    ----------
    path : str | Path
        File path to the saved Dataset pickle file.

    Returns
    -------
    Dataset
        The deserialized Dataset instance.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    pickle.UnpicklingError
        If the file cannot be deserialized as a valid pickle object.
    OSError
        If the file cannot be read due to permission or I/O errors.
    RuntimeError
        If an unexpected error occurs during deserialization.

    Examples
    --------
    >>> dataset = _load_dataset('data/my_dataset.pkl')
    >>> dataset = _load_dataset('data/my_dataset.pkl.gz')
    """
    try:
        # Convert path to Path object for cross-platform compatibility
        file_path: Path = Path(path)

        with open(file_path, "rb") as fh:
            return pickle.load(fh)

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to deserialize dataset from {path}: {e}")
    except OSError as e:
        raise OSError(f"Failed to read dataset from {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during dataset load: {e}")


def save_dataset_to_file(dataset: Dataset, compress: bool = False) -> None:
    """
    Save a Dataset object to disk in the project's standard serialized format.
    This function serves as a convenience wrapper around the lower-level
    `src.dataset_serializer._save_dataset` function. It handles the extraction
    of subject identification and tag generation, then constructs the appropriate
    file path before serializing the dataset.
    Args:
        dataset (Dataset): The dataset object to be persisted to disk.
    Returns:
        None
    Raises:
        ValueError: If the dataset is invalid or missing required fields.
        IOError: If file writing fails or the output directory is inaccessible.
    Example:
        >>> my_dataset = Dataset(subject_id="SUBJ_001", ...)
        >>> save_dataset_to_file(my_dataset, compress=True)
    """
    # Extract subject identifier and optional tags from the Dataset.
    subject_id: str = dataset.subject_id
    tags: tuple[str, ...] = (
        tuple()
    )  # TODO: determine tags based on dataset.stats or other metadata

    # Build the file path where the dataset will be saved.
    path: Path = _build_file_path(subject_id, tags)

    # Delegate to the lower-level saver (handles pickling & compression).
    _save_dataset(path, dataset)


def load_dataset_from_file(path: str | Path) -> Dataset:
    """
    Load and return a `Dataset` previously saved with :func:`save_dataset_to_file`.

    Parameters
    ----------
    path : str | Path
        Path to the saved dataset (can be gzip-compressed or uncompressed).

    Returns
    -------
    Dataset
        The loaded Dataset instance.

    Raises
    ------
    FileNotFoundError, pickle.UnpicklingError, OSError, RuntimeError
        Errors raised by the underlying `_load_dataset` are propagated.
    """
    # Delegate to the internal loader which handles both compressed and
    # uncompressed pickle files.
    ds: Dataset = _load_dataset(path)
    return ds


def load_all_datasets() -> tuple[Dataset, ...]:
    """
    Load all Dataset objects from the project's datasets directory.

    Scans the designated datasets folder for all pickle files matching the
    project's standard file extension and deserializes them into Dataset instances.

    Returns
    -------
    tuple[Dataset, ...]
        A tuple containing all loaded Dataset instances from the datasets directory.
        Returns an empty tuple if no dataset files are found.

    Raises
    ------
    FileNotFoundError
        If no dataset files are found in the datasets directory.

    Examples
    --------
    >>> all_datasets = load_all_datasets()
    >>> print(f"Loaded {len(all_datasets)} datasets")
    """
    # Retrieve all pickle files from the project's datasets directory
    dataset_files: list[Path] = folder.get_files_in_folder(
        folder=folder.get_datasets_dir(),
        file_extension=const.PICKLE_FILE_EXTENSION,
    )

    # Initialize an empty list to store deserialized Dataset instances
    datasets: list[Dataset] = []

    # Iterate over each dataset file and deserialize it
    for file in dataset_files:
        ds: Dataset = load_dataset_from_file(file)
        datasets.append(ds)

    if not datasets:
        raise FileNotFoundError("No dataset files found in the datasets directory.")

    # Convert the list to a tuple and return
    return tuple(datasets)


if __name__ == "__main__":

    # --- Example usage ---
    # This will be run as a smoke test to verify the save/load process works end-to-end
    import os

    sample_subject_id = "_sample_subject_001"  # Example subject ID for testing

    write_dataset: Dataset = Dataset(
        epochs=((np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])), 0),
        y_labels=("Wake", "N1", "N2", "N3", "REM"),
        subject_id=sample_subject_id,
        stats={"sampling_freq": 100.0, "num_bins": 10},
    )

    # read and write dataset to verify round-trip consistency
    print("Testing dataset save and load consistency:")
    try:
        print("Write dataset:", write_dataset)
        # Save and re-load the dataset to verify round-trip.
        save_dataset_to_file(write_dataset)

        # Build the path used by the saver and loader (same naming convention).
        data_set_path: Path = _build_file_path(sample_subject_id, tuple())

        # Load the dataset back from disk and compare.
        read_dataset: Dataset = load_dataset_from_file(data_set_path)
        print("Read dataset:", read_dataset)
        if _equals(write_dataset, read_dataset):
            print("SUCCESS: Dataset read and write are consistent.")
        else:
            raise ValueError("ERROR: Loaded dataset does not match original.")
    finally:
        if data_set_path.exists():
            os.remove(data_set_path)

    # writing an existing file name should not overwrite and should print "e"
    print(
        "\nTesting write with existing file (should print 'c' for created and 'e' for existing):"
    )
    try:
        save_dataset_to_file(write_dataset)  # First write (should create file)
        save_dataset_to_file(
            write_dataset
        )  # Second write (should detect existing file)
    finally:
        if data_set_path.exists():
            os.remove(data_set_path)

    # load all datasets in the datasets directory (will be empty after cleanup, so expect FileNotFoundError)
    print("\n\nTesting loading all datasets from the datasets directory:")
    all_datasets = load_all_datasets()
    print(f"Loaded {len(all_datasets)} datasets")
