"""
Folder utility module for managing project directory paths and file retrieval.

This module provides utilities for accessing key project directories (sleep data, reports)
and retrieving specific file types (EDF files, hypnogram files) from the sleep data directory.

Example:
    >>> from folders import get_sleep_data_dir, get_edf_files_in_sleep_data
    >>> sleep_dir = get_sleep_data_dir()
    >>> edf_files = get_edf_files_in_sleep_data()
    >>> print(f"Found {len(edf_files)} EDF files in {sleep_dir}")
"""

from pathlib import Path

from .constants import (
    EDF_FILE_SUFFIX,
    SLEEP_DATA_DIR,
    REPORTS_DIR,
    HYPNOGRAM_FILE_SUFFIX,
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


def get_reports_dir() -> Path:
    """
    Return the path to the reports directory.

    Retrieves the absolute path to the reports directory located within the project root.
    The directory must exist; if it does not, a FileNotFoundError is raised.

    Returns:
        Path: The absolute path object pointing to the reports directory.

    Raises:
        FileNotFoundError: If the reports directory does not exist at the expected location.

    Example:
        >>> reports_path = get_reports_dir()
        >>> reports_path.exists()
        True
    """
    path = _get_project_root() / REPORTS_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Reports directory not found: {path}. Create it manually."
        )
    return path.resolve()


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
    edf_files = list(sleep_dir.glob(f"*{EDF_FILE_SUFFIX}"))
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


if __name__ == "__main__":
    # Example usage
    print("Project root directory:", _get_project_root())
    print("Sleep data directory:", get_sleep_data_dir())
    print("Reports directory:", get_reports_dir())
    print("EDF files in sleep data directory:", get_edf_files_in_sleep_data())
    print(
        "Hypnogram files in sleep data directory:",
        get_hypnogram_files_in_sleep_data(),
    )
