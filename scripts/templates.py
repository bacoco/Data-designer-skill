#!/usr/bin/env python3
"""
Jinja2 templating utilities for prompt rendering.

Renders prompts with row data, supporting nested field access like {{ customer.first_name }}.
"""

import re
from typing import Any

from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError


class PromptRenderer:
    """
    Renders Jinja2 templates with row data.

    Supports:
    - Simple variables: {{ name }}
    - Nested access: {{ customer.first_name }}
    - Filters: {{ name | upper }}
    - Conditionals: {% if rating > 3 %}positive{% endif %}
    """

    def __init__(self):
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Add custom filters
        self.env.filters["json"] = self._json_filter

    @staticmethod
    def _json_filter(value: Any) -> str:
        """Convert value to JSON string."""
        import json
        return json.dumps(value)

    def render(self, template: str, row: dict[str, Any]) -> str:
        """
        Render a template with row data.

        Args:
            template: Jinja2 template string
            row: Dictionary of column values

        Returns:
            Rendered string

        Raises:
            TemplateSyntaxError: If template syntax is invalid
            UndefinedError: If referenced variable is missing
        """
        try:
            tpl = self.env.from_string(template)
            return tpl.render(**row)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {e}")
        except UndefinedError as e:
            raise ValueError(f"Missing variable in template: {e}")

    def get_variables(self, template: str) -> list[str]:
        """
        Extract variable names from a template.

        Returns top-level variable names (not nested fields).
        E.g., {{ customer.first_name }} returns "customer"
        """
        # Match {{ variable }} or {{ variable.field }} patterns
        pattern = r"\{\{\s*(\w+)(?:\.\w+)*\s*(?:\|[^}]+)?\}\}"
        matches = re.findall(pattern, template)
        return list(set(matches))

    def validate(self, template: str, available_columns: list[str]) -> tuple[bool, str | None]:
        """
        Validate that a template only references available columns.

        Args:
            template: Jinja2 template string
            available_columns: List of available column names

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check syntax
            self.env.from_string(template)
        except TemplateSyntaxError as e:
            return False, f"Invalid Jinja2 syntax: {e}"

        # Check variables
        required = self.get_variables(template)
        missing = [v for v in required if v not in available_columns]

        if missing:
            return False, f"Missing columns: {missing}"

        return True, None


def extract_dependencies(prompt: str) -> list[str]:
    """
    Extract column dependencies from a prompt template.

    Args:
        prompt: Jinja2 template string

    Returns:
        List of column names that the prompt depends on
    """
    renderer = PromptRenderer()
    return renderer.get_variables(prompt)


def render_prompt(prompt: str, row: dict[str, Any]) -> str:
    """
    Convenience function to render a prompt with row data.

    Args:
        prompt: Jinja2 template string
        row: Dictionary of column values

    Returns:
        Rendered prompt string
    """
    renderer = PromptRenderer()
    return renderer.render(prompt, row)


if __name__ == "__main__":
    # Test the renderer
    renderer = PromptRenderer()

    # Test data
    row = {
        "product_category": "Electronics",
        "rating": 4,
        "customer": {
            "first_name": "John",
            "last_name": "Doe",
            "city": "New York"
        }
    }

    # Simple template
    template1 = "This is a {{ product_category }} product with {{ rating }} stars."
    print(f"Template 1: {template1}")
    print(f"Rendered: {renderer.render(template1, row)}")
    print(f"Variables: {renderer.get_variables(template1)}")
    print()

    # Nested template
    template2 = "{{ customer.first_name }} from {{ customer.city }} rated this {{ rating }} stars."
    print(f"Template 2: {template2}")
    print(f"Rendered: {renderer.render(template2, row)}")
    print(f"Variables: {renderer.get_variables(template2)}")
    print()

    # Conditional template
    template3 = """
{% if rating >= 4 %}
Write a positive review
{% else %}
Write a negative review
{% endif %}
for {{ product_category }}.
""".strip()
    print(f"Template 3: {template3}")
    print(f"Rendered: {renderer.render(template3, row)}")
    print()

    # Validation
    valid, error = renderer.validate(template2, ["customer", "rating"])
    print(f"Valid: {valid}, Error: {error}")

    invalid, error = renderer.validate(template2, ["customer"])
    print(f"Valid (missing rating): {invalid}, Error: {error}")
