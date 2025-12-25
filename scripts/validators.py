#!/usr/bin/env python3
"""
Validators for synthetic data quality assurance.

Inspired by NVIDIA DataDesigner's validation system.
Supports regex, length, Python syntax, and JSON schema validation.
"""

import ast
import json
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str | None = None
    score: float | None = None  # Optional quality score 0-10
    details: dict[str, Any] | None = None


class BaseValidator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """Validate a single value."""
        pass

    def validate_batch(self, values: list[Any]) -> list[ValidationResult]:
        """Validate a batch of values."""
        return [self.validate(v) for v in values]


class RegexValidator(BaseValidator):
    """Validate string matches a regex pattern."""

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = re.compile(pattern, flags)

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                message=f"Expected string, got {type(value).__name__}"
            )

        match = self.pattern.match(value)
        if match:
            return ValidationResult(is_valid=True)
        else:
            return ValidationResult(
                is_valid=False,
                message=f"Value does not match pattern: {self.pattern.pattern}"
            )


class LengthValidator(BaseValidator):
    """Validate string length is within bounds."""

    def __init__(self, min_length: int = 0, max_length: int | None = None):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                message=f"Expected string, got {type(value).__name__}"
            )

        length = len(value)

        if length < self.min_length:
            return ValidationResult(
                is_valid=False,
                message=f"Length {length} is less than minimum {self.min_length}"
            )

        if self.max_length is not None and length > self.max_length:
            return ValidationResult(
                is_valid=False,
                message=f"Length {length} exceeds maximum {self.max_length}"
            )

        return ValidationResult(is_valid=True)


class PythonSyntaxValidator(BaseValidator):
    """Validate Python code syntax using AST."""

    def __init__(self, check_imports: bool = False):
        self.check_imports = check_imports

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                message=f"Expected string, got {type(value).__name__}"
            )

        try:
            tree = ast.parse(value)

            # Count statements for quality scoring
            stmt_count = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.stmt))

            return ValidationResult(
                is_valid=True,
                score=10.0,  # Perfect syntax score
                details={"statement_count": stmt_count}
            )

        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                score=0.0,
                details={"line": e.lineno, "offset": e.offset}
            )


class PythonLintValidator(BaseValidator):
    """
    Validate Python code using ruff linter.

    Requires ruff to be installed: pip install ruff
    """

    # Error type severity weights
    SEVERITY_WEIGHTS = {
        "fatal": 5,
        "error": 3,
        "warning": 1,
        "convention": 0.5,
        "refactor": 0.5,
    }

    def __init__(self, max_severity: str = "error"):
        """
        Args:
            max_severity: Maximum acceptable severity level.
                         One of: fatal, error, warning, convention, refactor
        """
        self.max_severity = max_severity
        self._check_ruff()

    def _check_ruff(self):
        """Check if ruff is available."""
        try:
            subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                check=True
            )
            self._has_ruff = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._has_ruff = False

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                message=f"Expected string, got {type(value).__name__}"
            )

        # First check basic syntax
        try:
            ast.parse(value)
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                message=f"Syntax error: {e.msg}",
                score=0.0
            )

        # If no ruff, just return syntax check result
        if not self._has_ruff:
            return ValidationResult(
                is_valid=True,
                score=10.0,
                message="Syntax valid (ruff not available for linting)"
            )

        # Run ruff
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(value)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", temp_path],
                capture_output=True,
                text=True
            )

            issues = json.loads(result.stdout) if result.stdout else []

            # Calculate score
            total_penalty = 0
            for issue in issues:
                code = issue.get("code", "")
                # Map ruff codes to severity
                if code.startswith("F"):
                    severity = "error"
                elif code.startswith("E"):
                    severity = "warning"
                else:
                    severity = "convention"
                total_penalty += self.SEVERITY_WEIGHTS.get(severity, 1)

            # Score from 0-10
            score = max(0, 10 - total_penalty)

            # Check if any issues exceed max severity
            severity_order = ["fatal", "error", "warning", "convention", "refactor"]
            max_idx = severity_order.index(self.max_severity)

            is_valid = True
            for issue in issues:
                code = issue.get("code", "")
                if code.startswith("F"):  # Fatal/Error
                    if max_idx < 1:
                        is_valid = False
                        break

            return ValidationResult(
                is_valid=is_valid,
                score=score,
                details={"issues": issues[:5]}  # First 5 issues
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)


class JsonSchemaValidator(BaseValidator):
    """Validate JSON data against a JSON Schema."""

    def __init__(self, schema: dict):
        self.schema = schema
        try:
            from jsonschema import Draft7Validator
            self._validator = Draft7Validator(schema)
            self._has_jsonschema = True
        except ImportError:
            self._has_jsonschema = False

    def validate(self, value: Any) -> ValidationResult:
        if not self._has_jsonschema:
            return ValidationResult(
                is_valid=True,
                message="jsonschema not installed, skipping validation"
            )

        # Parse if string
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Invalid JSON: {e}"
                )

        # Validate
        errors = list(self._validator.iter_errors(value))

        if not errors:
            return ValidationResult(is_valid=True, score=10.0)

        # Format error messages
        error_msgs = [f"{e.path}: {e.message}" if e.path else e.message for e in errors[:3]]

        return ValidationResult(
            is_valid=False,
            message="; ".join(error_msgs),
            score=max(0, 10 - len(errors)),
            details={"error_count": len(errors)}
        )


class CompositeValidator(BaseValidator):
    """Combine multiple validators."""

    def __init__(self, validators: list[BaseValidator], require_all: bool = True):
        """
        Args:
            validators: List of validators to apply
            require_all: If True, all must pass. If False, any must pass.
        """
        self.validators = validators
        self.require_all = require_all

    def validate(self, value: Any) -> ValidationResult:
        results = [v.validate(value) for v in self.validators]

        if self.require_all:
            # All must pass
            failed = [r for r in results if not r.is_valid]
            if failed:
                messages = [r.message for r in failed if r.message]
                return ValidationResult(
                    is_valid=False,
                    message="; ".join(messages),
                    score=min(r.score for r in results if r.score is not None) if any(r.score for r in results) else None
                )
            return ValidationResult(
                is_valid=True,
                score=sum(r.score for r in results if r.score) / len(results) if any(r.score for r in results) else None
            )
        else:
            # Any must pass
            passed = [r for r in results if r.is_valid]
            if passed:
                return ValidationResult(
                    is_valid=True,
                    score=max(r.score for r in passed if r.score is not None) if any(r.score for r in passed) else None
                )
            messages = [r.message for r in results if r.message]
            return ValidationResult(
                is_valid=False,
                message="; ".join(messages)
            )


def create_validator(config: dict) -> BaseValidator:
    """
    Create a validator from configuration.

    Args:
        config: Validator configuration with 'type' and 'params'

    Returns:
        Validator instance
    """
    validator_type = config.get("type")
    params = config.get("params", {})

    if validator_type == "regex":
        return RegexValidator(pattern=params.get("pattern", ".*"))

    elif validator_type == "length":
        return LengthValidator(
            min_length=params.get("min", 0),
            max_length=params.get("max")
        )

    elif validator_type == "python":
        return PythonSyntaxValidator()

    elif validator_type == "python_lint":
        return PythonLintValidator(max_severity=params.get("max_severity", "error"))

    elif validator_type == "json_schema":
        return JsonSchemaValidator(schema=params.get("schema", {}))

    else:
        raise ValueError(f"Unknown validator type: {validator_type}")


if __name__ == "__main__":
    # Test validators
    print("Testing validators...\n")

    # Regex
    print("=== Regex Validator ===")
    regex_v = RegexValidator(r"^\d{3}-\d{4}$")
    print(f"'123-4567': {regex_v.validate('123-4567')}")
    print(f"'abc': {regex_v.validate('abc')}")

    # Length
    print("\n=== Length Validator ===")
    length_v = LengthValidator(min_length=5, max_length=20)
    print(f"'hello': {length_v.validate('hello')}")
    print(f"'hi': {length_v.validate('hi')}")

    # Python syntax
    print("\n=== Python Syntax Validator ===")
    python_v = PythonSyntaxValidator()
    print(f"Valid code: {python_v.validate('def foo(): return 42')}")
    print(f"Invalid code: {python_v.validate('def foo( return')}")

    # JSON Schema
    print("\n=== JSON Schema Validator ===")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name"]
    }
    json_v = JsonSchemaValidator(schema)
    print(f"Valid: {json_v.validate({'name': 'John', 'age': 30})}")
    print(f"Missing name: {json_v.validate({'age': 30})}")

    # Composite
    print("\n=== Composite Validator ===")
    composite = CompositeValidator([
        LengthValidator(min_length=10),
        PythonSyntaxValidator()
    ])
    print(f"Valid: {composite.validate('def foo(): return 42')}")
    print(f"Too short: {composite.validate('x=1')}")

    print("\nAll validators working!")
