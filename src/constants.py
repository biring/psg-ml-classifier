"""
This module defines constants used throughout the sleep data analysis pipeline.
"""

# --- folder constants ---
DATASET_DIR = "data/datasets"
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

# --- training constants ---
EPOCH_LENGTH_SECONDS = (
    30  # Standard epoch length for sleep staging based on AASM guidelines
)
CHANNEL_SELECTION = "EEG Fpz-Cz"  # Selected channel for machine learning training
SLEEP_STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "Sleep",
    "Sleep stage 2": "Sleep",
    "Sleep stage 3": "Sleep",
    "Sleep stage 4": "Sleep",
    "Sleep stage R": "Sleep",
    "Sleep stage ?": "Invalid",
    "Movement time": "Invalid",
}  # Mapping of original sleep stage labels to simplified categories for machine learning training
