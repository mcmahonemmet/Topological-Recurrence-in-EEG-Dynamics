#!/usr/bin/env python3

"""
install_packages.py

Installs all Python dependencies required to run the EEG
Topological Recurrence analysis pipeline.

This script reads the requirements.txt file and installs
all listed packages using pip.

Usage:

    python install_packages.py
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Ensure Python version is sufficient."""
    required_major = 3
    required_minor = 10

    if sys.version_info < (required_major, required_minor):
        print(
            f"Python {required_major}.{required_minor}+ is required.\n"
            f"Current version: {sys.version}"
        )
        sys.exit(1)


def install_requirements():
    """Install packages listed in requirements.txt."""

    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("requirements.txt not found.")
        sys.exit(1)

    print("\nInstalling required Python packages...\n")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
    except subprocess.CalledProcessError:
        print("\nPackage installation failed.")
        sys.exit(1)

    print("\nAll dependencies installed successfully.")


def main():
    print("EEG Topological Recurrence Analysis")
    print("-----------------------------------")

    check_python_version()
    install_requirements()


if __name__ == "__main__":
    main()