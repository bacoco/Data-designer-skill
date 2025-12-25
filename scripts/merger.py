#!/usr/bin/env python3
"""
Merge batch files into final dataset.

Supports merging batches from multiple agents/directories and appending to an
existing output file so repeated runs accumulate data rather than overwrite it.

Usage:
    python merger.py --input batches/ --output final_dataset.csv
    python merger.py --input batches_a/ batches_b/ --output final_dataset.json
    python merger.py --input batches/ --output final_dataset.json --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from exporters import export_dataset, load_dataset


def load_batch(path: Path) -> list[dict[str, Any]]:
    """Load a batch file."""
    with open(path) as f:
        return json.load(f)


def merge_batches(batch_dir: Path) -> list[dict[str, Any]]:
    """
    Merge all batch files in a directory.

    Args:
        batch_dir: Directory containing batch_*.json files

    Returns:
        Combined list of all rows
    """
    # Find batch files
    batch_files = sorted(batch_dir.glob("batch_*.json"))

    if not batch_files:
        raise ValueError(f"No batch files found in {batch_dir}")

    # Merge
    all_rows = []
    for batch_file in batch_files:
        rows = load_batch(batch_file)
        all_rows.extend(rows)

    return all_rows


def merge_batch_directories(batch_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """
    Merge batches from multiple directories.

    Args:
        batch_dirs: Iterable of directories containing batch_*.json files

    Returns:
        Combined list of rows from all directories (sorted within each)
    """
    merged_rows: list[dict[str, Any]] = []

    for batch_dir in batch_dirs:
        try:
            dir_rows = merge_batches(batch_dir)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

        merged_rows.extend(dir_rows)

    if not merged_rows:
        raise ValueError("No batch files found in any provided directory")

    return merged_rows


def clean_rows(rows: list[dict[str, Any]], drop_columns: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Clean rows by removing internal fields and specified columns.

    Args:
        rows: List of row dictionaries
        drop_columns: Optional list of columns to drop

    Returns:
        Cleaned rows
    """
    drop_columns = drop_columns or []

    cleaned = []
    for row in rows:
        clean_row = {}
        for key, value in row.items():
            # Skip internal fields
            if key.startswith("_"):
                continue
            # Skip dropped columns
            if key in drop_columns:
                continue
            clean_row[key] = value
        cleaned.append(clean_row)

    return cleaned


def flatten_nested(rows: list[dict[str, Any]], separator: str = ".") -> list[dict[str, Any]]:
    """
    Flatten nested dictionaries in rows.

    E.g., {"customer": {"name": "John"}} -> {"customer.name": "John"}

    Args:
        rows: List of row dictionaries
        separator: Separator for nested keys

    Returns:
        Flattened rows
    """
    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return [flatten_dict(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(
        description="Merge batch files into final dataset"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        nargs="+",
        help="One or more directories containing batch files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json", "jsonl", "parquet"],
        help="Output format (auto-detected from extension if not specified)"
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        help="Columns to drop from output"
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten nested dictionaries"
    )
    parser.add_argument(
        "--separator",
        default=".",
        help="Separator for flattened keys (default: '.')"
    )

    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input]
    output_path = Path(args.output)

    # Merge batches
    print(f"Merging batches from: {', '.join(str(p) for p in input_dirs)}...")
    try:
        rows = merge_batch_directories(input_dirs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Merged {len(rows)} rows")

    # Append existing output if present
    if output_path.exists():
        try:
            existing_rows = load_dataset(output_path)
            print(f"Found existing output with {len(existing_rows)} rows, appending new rows")
            rows = existing_rows + rows
        except Exception as e:
            print(f"Warning: Could not load existing output ({e}); proceeding with new rows only")

    # Clean rows
    rows = clean_rows(rows, drop_columns=args.drop)

    # Flatten if requested
    if args.flatten:
        rows = flatten_nested(rows, separator=args.separator)
        print("Flattened nested dictionaries")

    # Determine format
    output_format = args.format
    if not output_format:
        suffix = output_path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".parquet": "parquet",
        }
        output_format = format_map.get(suffix, "csv")

    # Export
    print(f"Exporting to {output_path} ({output_format})...")
    export_dataset(rows, output_path, format=output_format)

    print(f"Done! Created {output_path}")

    # Print summary
    if rows:
        print(f"\nColumns: {list(rows[0].keys())}")
        print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
