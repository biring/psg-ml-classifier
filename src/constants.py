"""
This module defines constants used throughout the sleep data analysis pipeline.
"""

# --- folder constants ---
DATASET_DIR = "data/dataset"
REPORTS_DIR = "data/reports"
SLEEP_DATA_DIR = "data/physionet-sleep-data"


# --- file extension ---
PICKLE_FILE_EXTENSION = ".pkl"
PHYSIONET_FILE_EXTENSION = ".edf"

# --- file suffixes ---
PSG_FILE_SUFFIX = f"-PSG{PHYSIONET_FILE_EXTENSION}"
HYPNOGRAM_FILE_SUFFIX = f"-Hypnogram{PHYSIONET_FILE_EXTENSION}"

# --- physionet dataset constants ---
SUBJECT_ID_RANGE = range(0, 66)  # Subject IDs from 0 to 65
RECORDING_NIGHT = [2]  # Night 2 is considered more complete and consistent for analysis
