"""
Physionet dataset utilities.

This module provides a small convenience wrapper around MNE's
`mne.datasets.sleep_physionet.age.fetch_data` to download and summarize
Sleep-EDF (PhysioNet) recordings used in this project. It exposes the
`PhysioNetDataset` class which can be used to download a set of subjects
and produce a simple ASCII summary of the PSG and associated hypnogram
files stored in the project's `data/physionet-sleep-data` folder.

Example:
    from src import constants as const
    from src import folders as folder

    # Construct the helper with the project-local sleep data directory
    dataset = PhysioNetDataset(dataset_folder=folder.get_sleep_data_dir())
    dataset.download_dataset(subject_ids=const.SUBJECT_ID_RANGE,
                             recording_night=const.RECORDING_NIGHT)
    # Provide the PSG file suffix (project constant) when summarizing
    print(dataset.downloaded_summary(psg_file_suffix=const.PSG_FILE_SUFFIX))
"""

# --- library imports ---
import os
from pathlib import Path
import mne
from mne.datasets.sleep_physionet.age import fetch_data


class PhysioNetDataset:
    """
    Manage downloading and simple inspection of PhysioNet Sleep-EDF data.

    The class wraps MNE's fetching functionality and keeps a record of
    which subject IDs were successfully fetched and which failed.

    Attributes:
        dataset_folder (pathlib.Path): Path to the local dataset directory
            used for storing and scanning downloaded files.
        downloaded_subject_ids (list[int]): Subject IDs successfully downloaded
            (populated after calling ``download_dataset``). Initialized to an
            empty list by the constructor.
        failed_subject_ids (list[int]): Subject IDs that failed or were missing
            (populated after calling ``download_dataset``). Initialized to an
            empty list by the constructor.
    """

    def __init__(self, *, dataset_folder: Path):
        """
        Initialize the dataset helper.

        The constructor expects a path-like object pointing at the project's
        `data/physionet-sleep-data` folder (or equivalent). It configures MNE
        to use the project's `data` parent directory as the `MNE_DATA` root so
        downloaded files are stored inside the workspace.

        Args:
            dataset_folder (Path): Path to the local sleep data directory.
        """
        if not dataset_folder.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {dataset_folder}")

        self.dataset_folder: Path = dataset_folder

        # Configure MNE to store datasets in our project-local raw directory.
        # Use the parent `data` directory as MNE's root so MNE won't create
        # an inner `physionet-sleep-data` folder under a path that already
        # contains that same folder name (which would cause nesting).
        mne.set_config(
            "MNE_DATA", str(self.dataset_folder.parent).strip(), set_env=True
        )

    def download_dataset(self, *, subject_ids, recording_night):
        """
        Download recordings for a list of subjects.

        Uses ``mne.datasets.sleep_physionet.age.fetch_data`` for each subject
        in ``subject_ids``. The function records whether a subject was
        successfully fetched or failed/missing, stores results on the
        instance as ``downloaded_subject_ids`` and ``failed_subject_ids``,
        and prints a concise progress summary to stdout.

        Args:
            subject_ids (Iterable[int]): Sequence of numeric subject identifiers
                to attempt to download. May be any iterable of integers.
            recording_night (int): The recording night index passed to
                ``fetch_data`` (commonly 0 or 1 depending on dataset split).

        Notes:
            The method swallows exceptions raised by ``fetch_data`` and
            records the subject in ``failed_subject_ids`` so the caller can
            inspect which subjects did not download.
        """
        downloaded_subject_ids = []
        failed_subject_ids = []

        print("--- Starting dataset download ---")

        for n, subject_id in enumerate(subject_ids):
            print("#", end="")
            try:
                all_files = fetch_data(
                    subjects=[subject_id],
                    recording=recording_night,
                    on_missing="ignore",
                )

                if not all_files:
                    failed_subject_ids.append(subject_id)
                    continue

                downloaded_subject_ids.append(subject_id)

            except Exception:
                failed_subject_ids.append(subject_id)

        print("\n--- Dataset download complete ---")
        print(f"Total subjects checked: {n + 1}")
        print(f"Total subjects downloaded: {len(downloaded_subject_ids)}")
        print(f"Total subjects failed: {len(failed_subject_ids)}")
        print(f"Failed subject IDs: {failed_subject_ids}")

        self.downloaded_subject_ids = downloaded_subject_ids
        self.failed_subject_ids = failed_subject_ids

    def downloaded_summary(self, *, psg_file_suffix: str) -> str:
        """
        Create a human-readable ASCII summary of downloaded recordings.

        Scans the directory referred to by ``self.dataset_folder`` for files
        ending with the provided ``psg_file_suffix`` and attempts to match
        hypnogram files by subject prefix and the substring "Hypnogram". The
        returned string contains a simple table with one row per PSG
        recording and a final count of discovered subjects.

        Args:
            psg_file_suffix (str): The filename suffix used to identify PSG
                recordings (for example, ``"-PSG.edf"``). This value is used
                with ``str.endswith`` when scanning the dataset folder.

        Returns:
            str: Formatted table suitable for printing to a terminal.

        Raises:
            FileNotFoundError: If ``self.dataset_folder`` does not exist or
                cannot be listed. ``os.listdir`` will propagate the
                underlying OSError when the path is invalid or inaccessible.
        """
        # Get a list of all files in the sleep data directory
        folder_path: Path = self.dataset_folder
        all_files: list[str] = os.listdir(folder_path)

        # Filter and sort PSG files by PSG filename suffix (include all PSG recordings)
        psg_files: list[str] = sorted(
            [f for f in all_files if f.endswith(psg_file_suffix)], key=lambda x: x[:5]
        )

        # Counter for total subjects found
        subject_count: int = 0

        # List to accumulate log lines for the summary table
        log: list[str] = []

        # Add header row
        log.append("-" * 60)
        log.append(f"{'No':<5} {'Subject':<10} {'PSG':<20} {'Hypnogram':<20}")
        log.append("-" * 60)

        # Iterate through each PSG file and find matching Hypnogram file
        for psg in psg_files:
            # Extract subject ID from PSG filename (first 5 characters)
            subject_id: str = psg[:5]

            # Search for matching Hypnogram files with the same subject ID
            hypno_match: list[str] = [
                h for h in all_files if h.startswith(subject_id) and "Hypnogram" in h
            ]

            # Assign PSG filename
            psg_file_name: str = psg

            # Determine Hypnogram filename based on matches found
            if len(hypno_match) == 0:
                hypno_file_name: str = "MISSING"
            elif len(hypno_match) == 1:
                hypno_file_name: str = hypno_match[0]
            else:
                # Join multiple matching hypnogram files into a comma-separated string
                hypno_file_name: str = ", ".join(hypno_match)

            # Increment subject count and format row
            subject_count += 1
            row: str = (
                f"{subject_count:<5} {subject_id:<10} {psg_file_name:<20} {hypno_file_name:<20}"
            )
            log.append(row)

        # Add the total summary at the bottom
        total_line: str = f"Total Subjects Found: {subject_count}"
        log.append("-" * 60)
        log.append(total_line)

        # Return formatted summary as single string
        return "\n".join(log)


if __name__ == "__main__":
    # Example usage
    from src import constants as const
    from src import folders as folder

    dataset = PhysioNetDataset(dataset_folder=folder.get_sleep_data_dir())
    dataset.download_dataset(
        subject_ids=const.SUBJECT_ID_RANGE, recording_night=const.RECORDING_NIGHT
    )
    print(dataset.downloaded_summary(psg_file_suffix=const.PSG_FILE_SUFFIX))
