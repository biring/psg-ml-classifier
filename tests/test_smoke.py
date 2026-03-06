"""
Smoke test suite for validating all Python modules in the src/ directory.
This test script discovers and executes every .py file in the src/ directory as a module
using Python's `-m` flag. Each module is expected to have a smoke test that validates
its core functionality and raises an error if something is wrong.
The script can be used as a comprehensive smoke test to verify that all modules in src/
are free of import errors and syntax issues. It will fail on the first error encountered,
allowing developers to quickly identify which module(s) have problems.
Usage:
    python -m pytest tests/test_smoke.py
    or
    python tests/test_smoke.py
This module provides:
    - run_all_src_modules(): Function to discover and execute all src modules as subprocesses
This test script runs every .py file in `src/` as a module using `-m`.
Each module has a smoke test to ensure it executes without errors.
This script can be run as a smoke test to verify that all modules in `src/` are free of errors.
This test will fail on first error encountered, so it can be used to identify which module(s) have issues.
"""

import subprocess
import sys
from pathlib import Path


def run_all_src_modules() -> None:
    """
    Run every .py file in `src/` as a module using `-m`.

    Discovers all Python files in the src/ directory relative to this test file,
    constructs their module names, and executes each as a subprocess module.
    Raises RuntimeError on first module execution failure.

    Raises:
        FileNotFoundError: If src folder or Python files are not found.
        RuntimeError: If any module execution fails.
    """
    # Determine the project root directory (parent of tests/)
    project_root: Path = Path(__file__).resolve().parent.parent
    src_folder: Path = project_root / "src"

    # Verify that the src folder exists
    if not src_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {src_folder}")

    # Discover all Python files in src/ recursively
    python_files: list[Path] = sorted(src_folder.glob("**/*.py"))

    # Ensure at least one Python file was found
    if not python_files:
        raise FileNotFoundError(f"No Python files found in {src_folder}")

    # Execute each Python file as a module
    for file in python_files:
        # Build module name relative to the project root, e.g. src.feature_extractor
        rel: Path = file.relative_to(project_root).with_suffix("")
        module_name: str = rel.as_posix().replace("/", ".")

        # Display module execution progress
        print(f"\n{'-'*50}")
        print(f"Running module: {module_name} (from {file})")
        print("-" * 50)

        try:
            # Execute the module as a subprocess
            subprocess.run(
                [sys.executable, "-m", module_name], check=True, cwd=str(project_root)
            )
        except subprocess.CalledProcessError as e:
            # Raise an exception so callers (including the main block or test runner)
            # can detect the failure instead of continuing silently.
            raise RuntimeError(f"Error running {module_name}: {e}") from e

    # Display success message
    print(f"\n{'='*50}")
    print("SUCCESS: All modules ran without errors.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Run all modules and catch any errors
    # Each module is expected to have a smoke test that raises an error if something is wrong
    try:
        run_all_src_modules()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
