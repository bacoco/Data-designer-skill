#!/usr/bin/env python3
"""
Export utilities for synthetic datasets.

Supports CSV, JSON, JSONL, and Parquet formats.
"""

import csv
import json
from pathlib import Path
from typing import Any


def export_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    delimiter: str = ",",
    flatten_dicts: bool = True
) -> None:
    """
    Export data to CSV file.

    Args:
        rows: List of row dictionaries
        path: Output file path
        delimiter: CSV delimiter
        flatten_dicts: Convert nested dicts to JSON strings
    """
    if not rows:
        return

    # Process rows - convert complex types to strings
    processed_rows = []
    for row in rows:
        processed = {}
        for key, value in row.items():
            if isinstance(value, dict):
                if flatten_dicts:
                    processed[key] = json.dumps(value)
                else:
                    # Flatten nested dict
                    for k, v in value.items():
                        processed[f"{key}.{k}"] = v
            elif isinstance(value, list):
                processed[key] = json.dumps(value)
            else:
                processed[key] = value
        processed_rows.append(processed)

    # Get all columns
    columns = []
    for row in processed_rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    # Write CSV
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(processed_rows)


def export_json(
    rows: list[dict[str, Any]],
    path: str | Path,
    indent: int = 2
) -> None:
    """
    Export data to JSON file.

    Args:
        rows: List of row dictionaries
        path: Output file path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=indent, ensure_ascii=False, default=str)


def export_jsonl(
    rows: list[dict[str, Any]],
    path: str | Path
) -> None:
    """
    Export data to JSONL (JSON Lines) file.

    Args:
        rows: List of row dictionaries
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def export_parquet(
    rows: list[dict[str, Any]],
    path: str | Path
) -> None:
    """
    Export data to Parquet file.

    Requires pandas and pyarrow.

    Args:
        rows: List of row dictionaries
        path: Output file path
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet export: pip install pandas pyarrow")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Convert dict columns to JSON strings
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    df.to_parquet(path, index=False)


def export_dataset(
    rows: list[dict[str, Any]],
    path: str | Path,
    format: str = "csv",
    **kwargs
) -> None:
    """
    Export dataset to specified format.

    Args:
        rows: List of row dictionaries
        path: Output file path
        format: Output format (csv, json, jsonl, parquet)
        **kwargs: Format-specific options
    """
    format = format.lower()

    if format == "csv":
        export_csv(rows, path, **kwargs)
    elif format == "json":
        export_json(rows, path, **kwargs)
    elif format == "jsonl":
        export_jsonl(rows, path, **kwargs)
    elif format == "parquet":
        export_parquet(rows, path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """
    Load dataset from file.

    Args:
        path: Input file path

    Returns:
        List of row dictionaries
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    elif suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet: pip install pandas pyarrow")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


if __name__ == "__main__":
    # Test exports
    import tempfile

    test_data = [
        {
            "id": 1,
            "name": "John",
            "metadata": {"age": 30, "city": "NYC"},
            "tags": ["a", "b"]
        },
        {
            "id": 2,
            "name": "Jane",
            "metadata": {"age": 25, "city": "LA"},
            "tags": ["c"]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV
        csv_path = Path(tmpdir) / "test.csv"
        export_csv(test_data, csv_path)
        print(f"CSV exported to {csv_path}")
        with open(csv_path) as f:
            print(f.read())

        # Test JSON
        json_path = Path(tmpdir) / "test.json"
        export_json(test_data, json_path)
        print(f"\nJSON exported to {json_path}")

        # Test JSONL
        jsonl_path = Path(tmpdir) / "test.jsonl"
        export_jsonl(test_data, jsonl_path)
        print(f"JSONL exported to {jsonl_path}")

        # Load back
        loaded = load_dataset(csv_path)
        print(f"\nLoaded from CSV: {len(loaded)} rows")

        loaded = load_dataset(json_path)
        print(f"Loaded from JSON: {len(loaded)} rows")

    print("\nAll exports working!")
