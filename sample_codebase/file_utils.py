"""
File utilities for the sample codebase.
Provides helper functions for file I/O operations.
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileManager:
    """Manages file read/write operations with support for multiple formats."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Read and parse a JSON file.

        Args:
            filename: Name of the JSON file relative to base_dir.

        Returns:
            Parsed JSON data as a dict, or None if the file doesn't exist.
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, filename: str, data: Dict[str, Any], indent: int = 2) -> None:
        """Write data to a JSON file.

        Args:
            filename: Name of the output file relative to base_dir.
            data: The dictionary to serialize.
            indent: JSON indentation level.
        """
        filepath = self.base_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    def read_csv(self, filename: str) -> List[Dict[str, str]]:
        """Read a CSV file and return rows as a list of dicts.

        Args:
            filename: Name of the CSV file relative to base_dir.

        Returns:
            A list of dicts where each dict represents a row.
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def write_csv(self, filename: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
        """Write rows to a CSV file.

        Args:
            filename: Name of the output CSV file.
            rows: List of dicts to write.
            fieldnames: Column headers.
        """
        filepath = self.base_dir / filename
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching a glob pattern in the base directory.

        Args:
            pattern: Glob pattern to match (default: all files).

        Returns:
            List of matching file paths as strings.
        """
        return [str(p) for p in self.base_dir.glob(pattern) if p.is_file()]


def count_lines(filepath: str) -> int:
    """Count the number of lines in a text file.

    Args:
        filepath: Path to the file.

    Returns:
        Number of lines in the file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def find_files_by_extension(root_dir: str, extension: str) -> List[str]:
    """Recursively find all files with a given extension.

    Args:
        root_dir: Directory to search.
        extension: File extension to match (e.g., '.py').

    Returns:
        List of matching file paths.
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(extension):
                matches.append(os.path.join(dirpath, fname))
    return matches
