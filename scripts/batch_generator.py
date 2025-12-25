#!/usr/bin/env python3
"""
Batch data generator for synthetic datasets.

Generates data in batches:
1. First generates all sampler columns
2. Outputs seed data for LLM columns to be filled by Claude
3. Supports incremental batch generation

Usage:
    python batch_generator.py --schema schema.json --rows 100 --output batch.json
    python batch_generator.py --schema schema.json --rows 100 --batch-size 20 --output batches/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from samplers import SamplerRegistry, generate_samples
from schema_builder import DatasetSchema, ColumnType
from templates import PromptRenderer


def generate_sampler_data(schema: DatasetSchema, num_rows: int) -> list[dict[str, Any]]:
    """
    Generate data for all sampler columns.

    Args:
        schema: Dataset schema
        num_rows: Number of rows to generate

    Returns:
        List of row dictionaries with sampler column values
    """
    rows = [{} for _ in range(num_rows)]

    # Get generation order
    order = schema.get_generation_order()

    for col_name in order:
        col = schema.get_column(col_name)
        if col is None or not col.type.is_sampler:
            continue

        # Generate samples
        samples = generate_samples(
            {"type": col.type.value, "params": col.params},
            n=num_rows,
            seed=schema.seed
        )

        # Add to rows
        for i, sample in enumerate(samples):
            rows[i][col_name] = sample

    return rows


def prepare_llm_prompts(
    schema: DatasetSchema,
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Prepare LLM prompts for each row.

    Renders Jinja2 templates with row data and returns
    the prompts that need to be filled.

    Args:
        schema: Dataset schema
        rows: Rows with sampler data

    Returns:
        List of row dicts with _llm_prompts field added
    """
    renderer = PromptRenderer()
    order = schema.get_generation_order()

    for row in rows:
        row["_llm_prompts"] = {}

        for col_name in order:
            col = schema.get_column(col_name)
            if col is None or not col.type.is_llm:
                continue

            # Render prompt with current row data
            try:
                rendered_prompt = renderer.render(col.prompt, row)
                system_prompt = None
                if col.system_prompt:
                    system_prompt = renderer.render(col.system_prompt, row)

                row["_llm_prompts"][col_name] = {
                    "prompt": rendered_prompt,
                    "system_prompt": system_prompt,
                    "type": col.type.value,
                    "schema": col.schema,
                    "code_lang": col.code_lang,
                }
            except Exception as e:
                row["_llm_prompts"][col_name] = {
                    "error": str(e)
                }

    return rows


def generate_batch(
    schema: DatasetSchema,
    num_rows: int,
    include_llm_prompts: bool = True
) -> list[dict[str, Any]]:
    """
    Generate a batch of data.

    Args:
        schema: Dataset schema
        num_rows: Number of rows to generate
        include_llm_prompts: Whether to include LLM prompts for Claude to fill

    Returns:
        List of row dictionaries
    """
    # Generate sampler data
    rows = generate_sampler_data(schema, num_rows)

    # Prepare LLM prompts if needed
    if include_llm_prompts and schema.get_llm_columns():
        rows = prepare_llm_prompts(schema, rows)

    return rows


def save_batch(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save batch to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def generate_batches(
    schema: DatasetSchema,
    total_rows: int,
    batch_size: int,
    output_dir: str | Path
) -> list[Path]:
    """
    Generate multiple batches.

    Args:
        schema: Dataset schema
        total_rows: Total number of rows to generate
        batch_size: Rows per batch
        output_dir: Directory to save batches

    Returns:
        List of batch file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_paths = []
    rows_generated = 0
    batch_num = 1

    while rows_generated < total_rows:
        rows_in_batch = min(batch_size, total_rows - rows_generated)
        batch = generate_batch(schema, rows_in_batch)

        batch_path = output_dir / f"batch_{batch_num:03d}.json"
        save_batch(batch, batch_path)
        batch_paths.append(batch_path)

        rows_generated += rows_in_batch
        batch_num += 1

    return batch_paths


def print_preview(rows: list[dict[str, Any]], max_rows: int = 3) -> None:
    """Print a preview of generated data."""
    print("\n" + "=" * 60)
    print("PREVIEW")
    print("=" * 60)

    for i, row in enumerate(rows[:max_rows]):
        print(f"\n--- Row {i + 1} ---")
        for key, value in row.items():
            if key == "_llm_prompts":
                print(f"\nLLM Prompts to fill:")
                for col_name, prompt_info in value.items():
                    if "error" in prompt_info:
                        print(f"  {col_name}: ERROR - {prompt_info['error']}")
                    else:
                        prompt_preview = prompt_info["prompt"][:100]
                        if len(prompt_info["prompt"]) > 100:
                            prompt_preview += "..."
                        print(f"  {col_name}: {prompt_preview}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data batches from schema"
    )
    parser.add_argument(
        "--schema", "-s",
        required=True,
        help="Path to schema JSON/YAML file"
    )
    parser.add_argument(
        "--rows", "-n",
        type=int,
        default=10,
        help="Number of rows to generate"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Rows per batch (if not set, single batch)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file or directory"
    )
    parser.add_argument(
        "--no-llm-prompts",
        action="store_true",
        help="Don't include LLM prompts (sampler data only)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print preview of generated data"
    )

    args = parser.parse_args()

    # Load schema
    try:
        schema = DatasetSchema.from_file(args.schema)
    except Exception as e:
        print(f"Error loading schema: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate schema
    valid, errors = schema.validate()
    if not valid:
        print("Schema validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating {args.rows} rows for '{schema.name}'...")
    print(f"Columns: {[col.name for col in schema.columns]}")
    print(f"Generation order: {schema.get_generation_order()}")

    # Generate data
    if args.batch_size:
        # Multiple batches
        batch_paths = generate_batches(
            schema,
            args.rows,
            args.batch_size,
            args.output
        )
        print(f"\nGenerated {len(batch_paths)} batches in {args.output}/")
        for path in batch_paths:
            print(f"  - {path.name}")

        # Preview first batch
        if args.preview:
            with open(batch_paths[0]) as f:
                preview_rows = json.load(f)
            print_preview(preview_rows)

    else:
        # Single batch
        rows = generate_batch(
            schema,
            args.rows,
            include_llm_prompts=not args.no_llm_prompts
        )
        save_batch(rows, args.output)
        print(f"\nGenerated {len(rows)} rows to {args.output}")

        if args.preview:
            print_preview(rows)

    # Summary
    llm_cols = schema.get_llm_columns()
    if llm_cols and not args.no_llm_prompts:
        print(f"\nLLM columns requiring Claude generation: {[c.name for c in llm_cols]}")
        print("Use the prompts in '_llm_prompts' field for each row.")


if __name__ == "__main__":
    main()
