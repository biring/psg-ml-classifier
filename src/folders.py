"""
Folder utility module for managing project directory paths and file retrieval.

This module provides utilities for accessing key project directories (sleep data, datasets, artifacts)
and retrieving specific file types (PSG/EDF files, hypnogram files) from the sleep data directory.
It includes functions for directory validation, file discovery, EDF-hypnogram pair matching, and
subject-based file lookup.

Example:
    >>> from folders import get_sleep_data_dir, get_edf_file_pairs
    >>> sleep_dir = get_sleep_data_dir()
    >>> pairs = get_edf_file_pairs()
    >>> subject_id, psg_path, hyp_path = pairs[0]
    >>> print(f"Subject {subject_id}: {psg_path.name}, {hyp_path.name}")
"""

from pathlib import Path

from .constants import (
    # folder constants
    ARTIFACTS_DIR,
    DATASET_DIR,
    SLEEP_DATA_DIR,
    # file suffixes
    HYPNOGRAM_FILE_SUFFIX,
    PSG_FILE_SUFFIX,
)


def _get_project_root() -> Path:
    """
    Return the project root directory.

    Retrieves the absolute path to the project root directory (top-level directory
    containing this file).

    Returns:
        Path: The absolute path object pointing to the project root directory.

    Example:
        >>> root = _get_project_root()
        >>> root.exists()
        True
    """
    return Path(__file__).parent.parent.resolve()


def get_sleep_data_dir() -> Path:
    """
    Return the path to the sleep data directory.

    Retrieves the absolute path to the sleep data directory located within the project root.
    The directory must exist; if it does not, a FileNotFoundError is raised.

    Returns:
        Path: The absolute path object pointing to the sleep data directory.

    Raises:
        FileNotFoundError: If the sleep data directory does not exist at the expected location.

    Example:
        >>> sleep_path = get_sleep_data_dir()
        >>> sleep_path.exists()
        True
    """
    path = _get_project_root() / SLEEP_DATA_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Sleep data directory not found: {path}. Create it manually."
        )
    return path.resolve()


def get_datasets_dir() -> Path:
    """
    Return the path to the datasets directory.

    Retrieves the absolute path to the datasets directory located within the project root.
    The directory will be created if it does not exist.

    Returns:
        Path: The absolute path object pointing to the datasets directory.

    Example:
        >>> datasets_path = get_datasets_dir()
        >>> datasets_path.exists()
        True
    """
    path = _get_project_root() / DATASET_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Datasets directory not found: {path}. Create it manually."
        )
    return path.resolve()


def get_artifacts_dir() -> Path:
    """
    Return the path to the artifacts directory.

    Retrieves the absolute path to the artifacts directory located within the project root.
    The directory must exist; if it does not, a FileNotFoundError is raised.

    Returns:
        Path: The absolute path object pointing to the artifacts directory.

    Raises:
        FileNotFoundError: If the artifacts directory does not exist at the expected location.

    Example:
        >>> artifacts_path = get_artifacts_dir()
        >>> artifacts_path.exists()
        True
    """
    path = _get_project_root() / ARTIFACTS_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found: {path}. Create it manually."
        )
    return path.resolve()


def get_files_in_folder(folder: Path, file_extension: str) -> tuple[Path, ...]:
    """
    Retrieve all files with a specified extension from a given folder.

    Searches the specified folder for files matching the given extension.
    The folder must exist and be a valid directory containing at least one
    matching file.

    Args:
        folder (Path): The path to the folder to search.
        file_extension (str): The file extension to match (e.g., ".edf", ".pkl").
            Must start with a dot and cannot be empty.

    Returns:
        tuple[Path, ...]: A tuple of Path objects for files matching the extension.

    Raises:
        FileNotFoundError: If the folder does not exist or no files with the
            specified extension are found.
        NotADirectoryError: If the folder path points to a file, not a directory.
        ValueError: If the file extension is empty or does not start with a dot.

    Example:
        >>> pkl_files = get_files_in_folder(get_datasets_dir(), ".pkl")
        >>> len(pkl_files) > 0
        True
    """

    # Verify the folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}. Create it manually.")

    # Ensure the path is a directory, not a file
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory but found a file: {folder}")

    # Validate file extension is not empty
    if not file_extension:
        raise ValueError("File extension cannot be empty.")

    # Validate file extension starts with a dot
    if not file_extension.startswith("."):
        raise ValueError(
            f"File extension should start with a dot (e.g., '.edf'): {file_extension}"
        )

    # Search for files whose suffix exactly matches the given extension
    files: list[Path] = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix == file_extension
    ]

    # Ensure at least one matching file was found
    if not files:
        raise FileNotFoundError(
            f"No files with suffix '{file_extension}' found in folder: {folder}"
        )

    return tuple(files)


def get_edf_files_in_sleep_data() -> list[Path]:
    """
    Return a list of EDF file paths in the sleep data directory.

    Retrieves all EDF files from the sleep data directory. The directory must exist
    and contain at least one EDF file; if not, a FileNotFoundError is raised.

    Returns:
        list[Path]: A list of Path objects pointing to EDF files in the sleep data directory.

    Raises:
        FileNotFoundError: If no EDF files are found in the sleep data directory.

    Example:
        >>> edf_files = get_edf_files_in_sleep_data()
        >>> len(edf_files) > 0
        True
    """
    sleep_dir = get_sleep_data_dir()
    edf_files = list(sleep_dir.glob(f"*{PSG_FILE_SUFFIX}"))
    if not edf_files:
        raise FileNotFoundError(
            f"No EDF files found in sleep data directory: {sleep_dir}"
        )
    return edf_files


def get_hypnogram_files_in_sleep_data() -> list[Path]:
    """
    Return a list of hypnogram file paths in the sleep data directory.

    Retrieves all hypnogram files from the sleep data directory. The directory must exist
    and contain at least one hypnogram file; if not, a FileNotFoundError is raised.

    Returns:
        list[Path]: A list of Path objects pointing to hypnogram files in the sleep data directory.

    Raises:
        FileNotFoundError: If no hypnogram files are found in the sleep data directory.

    Example:
        >>> hypno_files = get_hypnogram_files_in_sleep_data()
        >>> len(hypno_files) > 0
        True
    """
    sleep_dir = get_sleep_data_dir()
    hypno_files = list(sleep_dir.glob(f"*{HYPNOGRAM_FILE_SUFFIX}"))
    if not hypno_files:
        raise FileNotFoundError(
            f"No hypnogram files found in sleep data directory: {sleep_dir}"
        )
    return hypno_files


def get_edf_file_by_subject_id(subject: str) -> Path:
    """
    Retrieves the full file path to the EDF file for a given subject ID.

    We expect the subject_id to be part of the file name.

    Args:
        subject_id (str): The unique identifier of the subject whose EDF file is to be retrieved.
    Returns:
        Path: The full file path to the EDF file corresponding to the subject ID.
    Raises:
        FileNotFoundError: If no EDF file matching the subject ID is found in the sleep data directory.
    Example:
        >>> file_path = get_edf_file_by_subject_id("subject_001")
        >>> print(file_path)
        Path('/path/to/sleep/data/subject_001_sleep.edf')
    """
    edf_files = get_edf_files_in_sleep_data()
    for edf_file_name in edf_files:
        if subject in edf_file_name.name:
            return edf_file_name

    raise FileNotFoundError(
        f"EDF file for subject '{subject}' not found in sleep data directory: {get_sleep_data_dir()}"
    )


def get_edf_file_pairs() -> tuple[str, Path, Path]:
    """
    Retrieve paired PSG and hypnogram file paths from the sleep data directory.

    Searches the sleep data directory for PSG (EDF) files and their corresponding
    hypnogram files, returning them as matched pairs. The function ensures that
    the number of PSG files matches the number of hypnogram files.

    Returns:
        tuple[str, Path, Path]: A tuple containing the subject ID,
        the path to the PSG file, and the path to the corresponding hypnogram file.

    Raises:
        FileNotFoundError: If no PSG files are found or if the count of PSG files
            does not match the count of hypnogram files in the sleep data directory.

    Example:
        >>> pairs = get_edf_file_pairs()
        >>> subject_id, psg_path, hyp_path = pairs[0]
        >>> print(subject_id, psg_path, hyp_path)
    """
    # Accumulate matched (subject_id, (psg_path, hyp_path)) pairs
    pairs: list[tuple[str, tuple[Path, Path]]] = []

    # Get the sleep data directory path
    sleep_data_folder: Path = get_sleep_data_dir()

    # Retrieve all PSG and hypnogram files from the sleep data directory
    psg_files: list[Path] = list(sleep_data_folder.glob(f"*{PSG_FILE_SUFFIX}"))
    hypno_files: list[Path] = list(sleep_data_folder.glob(f"*{HYPNOGRAM_FILE_SUFFIX}"))

    # Get file counts for validation
    psg_file_count: int = len(psg_files)
    hypno_file_count: int = len(hypno_files)

    # Ensure at least one PSG file exists
    if psg_file_count == 0:
        raise FileNotFoundError(
            f"No PSG files found in sleep data directory: {sleep_data_folder}"
        )

    # Ensure PSG and hypnogram file counts match
    if psg_file_count != hypno_file_count:
        raise FileNotFoundError(
            f"Mismatch in PSG and Hypnogram file counts in sleep data directory: {sleep_data_folder}. "
            f"Found {psg_file_count} PSG files and {hypno_file_count} Hypnogram files."
        )

    # Iterate through sorted PSG files to find matching hypnogram files
    for psg in sorted(psg_files):
        # Extract subject ID from the first 5 characters of the filename
        subject_id: str = psg.name[:5]

        # Find the corresponding hypnogram file for this subject
        hyp: Path = next(
            sleep_data_folder.glob(f"{subject_id}*{HYPNOGRAM_FILE_SUFFIX}"), None
        )
        if hyp:
            # Add the pair of paths to the results
            pairs.append((subject_id, psg, hyp))

    return tuple(pairs)


if __name__ == "__main__":
    # --- DEMO / SMOKE TEST ---
    # This block is for demonstration purposes and may be used as a smoke test.
    # It will print the project directory structure and file statistics.

    # Print project directory structure header
    print("-" * 60)
    print("PROJECT DIRECTORY STRUCTURE")
    print("-" * 60)
    # Display paths to key project directories
    print(f"Project root: {_get_project_root()}")
    print(f"Artifacts: {get_artifacts_dir()}")
    print(f"Datasets: {get_datasets_dir()}")
    print(f"Sleep data: {get_sleep_data_dir()}")

    # Print file statistics header
    print("-" * 60)
    print("FILE STATISTICS")
    print("-" * 60)
    # Retrieve all EDF, hypnogram files, and matched pairs
    edf_files = get_edf_files_in_sleep_data()
    hypno_files = get_hypnogram_files_in_sleep_data()
    pickle_files = get_files_in_folder(get_datasets_dir(), ".pkl")
    # Display file counts
    print(f"EDF files found: {len(edf_files)}")
    print(f"Hypnogram files found: {len(hypno_files)}")
    print(f"Pickle files found in datasets: {len(pickle_files)}")

    # Print sample files header
    print("-" * 60)
    print("SAMPLE FILES")
    print("-" * 60)
    # Get a sample subject ID and corresponding PSG and hypnogram file paths
    subject_id, psg_path, hyp_path = get_edf_file_pairs()[0]
    # Display sample filenames from retrieved collections
    print(f"Paired subject ID: {subject_id}")
    print(f"Paired PSG: {psg_path.name}")
    print(f"Paired Hypnogram: {hyp_path.name}")
    print(f"Pickle file sample: {pickle_files[0].name}")
