#!/usr/bin/env python3
"""
Schema parser and dependency graph builder.

Parses dataset schema JSON/YAML and determines column generation order
using topological sort based on dependencies.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from templates import extract_dependencies


class ColumnType(str, Enum):
    """Types of columns that can be generated."""

    # Statistical samplers
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    BERNOULLI = "bernoulli"
    POISSON = "poisson"
    DATETIME = "datetime"
    PERSON = "person"
    UUID = "uuid"

    # LLM-generated
    LLM_TEXT = "llm_text"
    LLM_CODE = "llm_code"
    LLM_STRUCTURED = "llm_structured"
    LLM_JUDGE = "llm_judge"

    # Derived
    EXPRESSION = "expression"

    @property
    def is_sampler(self) -> bool:
        """Check if this is a statistical sampler type."""
        return self in [
            ColumnType.CATEGORY,
            ColumnType.SUBCATEGORY,
            ColumnType.UNIFORM,
            ColumnType.GAUSSIAN,
            ColumnType.BERNOULLI,
            ColumnType.POISSON,
            ColumnType.DATETIME,
            ColumnType.PERSON,
            ColumnType.UUID,
        ]

    @property
    def is_llm(self) -> bool:
        """Check if this requires LLM generation."""
        return self in [
            ColumnType.LLM_TEXT,
            ColumnType.LLM_CODE,
            ColumnType.LLM_STRUCTURED,
            ColumnType.LLM_JUDGE,
        ]


@dataclass
class ColumnConfig:
    """Configuration for a single column."""

    name: str
    type: ColumnType
    params: dict[str, Any] = field(default_factory=dict)

    # LLM-specific
    prompt: str | None = None
    system_prompt: str | None = None
    schema: dict | None = None  # For structured output
    code_lang: str | None = None  # For code generation

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Validation
    validators: list[dict] = field(default_factory=list)

    # Output
    drop: bool = False  # Drop from final output (intermediate column)

    @property
    def required_columns(self) -> list[str]:
        """Get all columns this column depends on."""
        deps = set(self.depends_on)

        # Extract from prompt templates
        if self.prompt:
            deps.update(extract_dependencies(self.prompt))
        if self.system_prompt:
            deps.update(extract_dependencies(self.system_prompt))

        # Subcategory depends on parent
        if self.type == ColumnType.SUBCATEGORY and "category" in self.params:
            deps.add(self.params["category"])

        return list(deps)


@dataclass
class DatasetSchema:
    """Complete dataset schema."""

    name: str
    description: str = ""
    columns: list[ColumnConfig] = field(default_factory=list)
    seed: int | None = None

    # Output settings
    output_format: str = "csv"
    output_filename: str = "dataset"
    output_directory: str = "/mnt/user-data/outputs"

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSchema":
        """Create schema from dictionary."""
        columns = []
        for col_data in data.get("columns", []):
            col_type = ColumnType(col_data.get("type"))
            columns.append(
                ColumnConfig(
                    name=col_data.get("name"),
                    type=col_type,
                    params=col_data.get("params", {}),
                    prompt=col_data.get("prompt"),
                    system_prompt=col_data.get("system_prompt"),
                    schema=col_data.get("schema"),
                    code_lang=col_data.get("code_lang"),
                    depends_on=col_data.get("depends_on", []),
                    validators=col_data.get("validators", []),
                    drop=col_data.get("drop", False),
                )
            )

        output = data.get("output", {})

        return cls(
            name=data.get("name", "dataset"),
            description=data.get("description", ""),
            columns=columns,
            seed=data.get("seed"),
            output_format=output.get("format", "csv"),
            output_filename=output.get("filename", "dataset"),
            output_directory=output.get("directory", "/mnt/user-data/outputs"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetSchema":
        """Load schema from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetSchema":
        """Load schema from YAML file."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetSchema":
        """Load schema from file (auto-detect format)."""
        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "columns": [
                {
                    "name": col.name,
                    "type": col.type.value,
                    "params": col.params,
                    **({"prompt": col.prompt} if col.prompt else {}),
                    **({"system_prompt": col.system_prompt} if col.system_prompt else {}),
                    **({"schema": col.schema} if col.schema else {}),
                    **({"code_lang": col.code_lang} if col.code_lang else {}),
                    **({"depends_on": col.depends_on} if col.depends_on else {}),
                    **({"validators": col.validators} if col.validators else {}),
                    **({"drop": col.drop} if col.drop else {}),
                }
                for col in self.columns
            ],
            "output": {
                "format": self.output_format,
                "filename": self.output_filename,
                "directory": self.output_directory,
            },
        }

    def to_json(self, path: str | Path) -> None:
        """Save schema to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: str | Path) -> None:
        """Save schema to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_column(self, name: str) -> ColumnConfig | None:
        """Get column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_generation_order(self) -> list[str]:
        """
        Get topologically sorted order for column generation.

        Sampler columns come first (no dependencies),
        then LLM columns in dependency order.
        """
        # Build dependency graph
        graph: dict[str, set[str]] = {}
        for col in self.columns:
            graph[col.name] = set(col.required_columns)

        # Kahn's algorithm for topological sort
        in_degree = {name: 0 for name in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Start with nodes that have no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for name, deps in graph.items():
                if node in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0 and name not in result and name not in queue:
                        queue.append(name)

        # Check for cycles
        if len(result) != len(graph):
            remaining = set(graph.keys()) - set(result)
            raise ValueError(f"Circular dependency detected: {remaining}")

        return result

    def get_sampler_columns(self) -> list[ColumnConfig]:
        """Get all sampler columns."""
        return [col for col in self.columns if col.type.is_sampler]

    def get_llm_columns(self) -> list[ColumnConfig]:
        """Get all LLM columns."""
        return [col for col in self.columns if col.type.is_llm]

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the schema.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for duplicate column names
        names = [col.name for col in self.columns]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            errors.append(f"Duplicate column names: {set(duplicates)}")

        # Check dependencies exist
        for col in self.columns:
            for dep in col.required_columns:
                if dep not in names:
                    errors.append(f"Column '{col.name}' depends on unknown column '{dep}'")

        # Check for circular dependencies
        try:
            self.get_generation_order()
        except ValueError as e:
            errors.append(str(e))

        # Check LLM columns have prompts
        for col in self.get_llm_columns():
            if not col.prompt:
                errors.append(f"LLM column '{col.name}' missing prompt")

        return len(errors) == 0, errors


def load_schema(path: str) -> DatasetSchema:
    """Convenience function to load schema from file."""
    return DatasetSchema.from_file(path)


if __name__ == "__main__":
    # Example schema
    schema_dict = {
        "name": "product_reviews",
        "description": "Customer reviews dataset",
        "seed": 42,
        "columns": [
            {
                "name": "product_category",
                "type": "category",
                "params": {
                    "values": ["Electronics", "Clothing", "Books"],
                    "weights": [0.4, 0.35, 0.25]
                }
            },
            {
                "name": "rating",
                "type": "uniform",
                "params": {"low": 1, "high": 5, "dtype": "int"}
            },
            {
                "name": "customer",
                "type": "person",
                "params": {
                    "fields": ["first_name", "last_name", "city"],
                    "age_range": [18, 65]
                }
            },
            {
                "name": "review_text",
                "type": "llm_text",
                "prompt": "Write a {{ rating }}-star review for a {{ product_category }} product by {{ customer.first_name }}.",
                "depends_on": ["rating", "product_category", "customer"]
            }
        ],
        "output": {
            "format": "csv",
            "filename": "reviews"
        }
    }

    schema = DatasetSchema.from_dict(schema_dict)
    print(f"Schema: {schema.name}")
    print(f"Columns: {[col.name for col in schema.columns]}")
    print(f"Generation order: {schema.get_generation_order()}")

    valid, errors = schema.validate()
    print(f"Valid: {valid}")
    if errors:
        print(f"Errors: {errors}")

    # Save to file
    schema.to_json("test_schema.json")
    print("\nSaved to test_schema.json")
