# Data Designer Skill

A Claude Code skill for generating high-quality synthetic datasets using statistical samplers and Claude's native LLM capabilities. No external API keys required.

## Overview

Data Designer combines:
- **Statistical Samplers** - Python scripts for distributions, categories, personas (no LLM needed)
- **LLM Generation** - Claude directly generates text, code, and structured data
- **Iterative Workflow** - Preview → Refine → Generate at scale
- **File-Based State** - Schema and batch files persist across iterations

Inspired by [NVIDIA NeMo DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) but adapted for Claude Code's skill architecture.

## Installation

1. Copy this skill to your Claude Code skills directory:
   ```bash
   cp -r Data-designer-skill ~/.claude/skills/data-designer
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy scipy faker pyyaml jinja2 pandas pyarrow
   ```

## Quick Start

1. **Ask Claude to generate a dataset:**
   ```
   Generate 50 product reviews with ratings 1-5
   ```

2. **Claude will:**
   - Create a schema file defining columns
   - Generate preview data for approval
   - Produce the full dataset after confirmation

## Samplers

Statistical samplers generate seed data without LLM calls:

| Sampler | Description | Example |
|---------|-------------|---------|
| `category` | Weighted random choice | `["A", "B", "C"]` with weights |
| `subcategory` | Hierarchical based on parent | `Electronics → [Phones, Laptops]` |
| `uniform` | Uniform distribution | Ratings 1-5 |
| `gaussian` | Normal distribution | Prices ~ N(50, 15) |
| `bernoulli` | Binary with probability | 70% verified users |
| `poisson` | Poisson distribution | Event counts |
| `datetime` | Random dates in range | 2024-01-01 to 2024-12-31 |
| `person` | Synthetic personas | Name, age, email, city |
| `uuid` | Unique identifiers | `ID-550e8400...` |

## LLM Column Types

Claude generates content for these column types:

| Type | Description |
|------|-------------|
| `llm_text` | Free-form text generation |
| `llm_code` | Code with syntax validation |
| `llm_structured` | JSON matching schema |
| `llm_judge` | Quality scoring |

## Schema Format

Define datasets in JSON or YAML:

```yaml
name: product_reviews
description: Customer reviews dataset
seed: 42

columns:
  - name: product_category
    type: category
    params:
      values: ["Electronics", "Books", "Clothing"]
      weights: [0.4, 0.35, 0.25]

  - name: rating
    type: uniform
    params:
      low: 1
      high: 5
      dtype: int

  - name: customer
    type: person
    params:
      fields: ["first_name", "age", "city"]

  - name: review_text
    type: llm_text
    prompt: |
      Write a {{ rating }}-star review for a {{ product_category }}
      product by {{ customer.first_name }} from {{ customer.city }}.
    depends_on:
      - rating
      - product_category
      - customer

output:
  format: csv
  filename: reviews
```

## Jinja2 Templating

Reference other columns in prompts:

```
Write a {{ rating }}-star review for {{ product_name }}
by {{ customer.first_name }} from {{ customer.city }}.
```

Supports:
- Simple variables: `{{ name }}`
- Nested access: `{{ customer.first_name }}`
- Conditionals: `{% if rating > 3 %}positive{% endif %}`
- Filters: `{{ name | upper }}`

## Scripts

### batch_generator.py

Generate data batches:

```bash
# Single batch
python scripts/batch_generator.py \
  --schema examples/product_reviews.json \
  --rows 50 \
  --output preview.json \
  --preview

# Multiple batches
python scripts/batch_generator.py \
  --schema examples/product_reviews.json \
  --rows 1000 \
  --batch-size 100 \
  --output batches/
```

### merger.py

Merge batches into final dataset:

```bash
python scripts/merger.py \
  --input batches/ \
  --output final_dataset.csv \
  --flatten
```

### validators.py

Validate generated data:

```python
from validators import PythonSyntaxValidator, RegexValidator

# Validate Python code
validator = PythonSyntaxValidator()
result = validator.validate("def foo(): return 42")
print(result.is_valid)  # True

# Validate patterns
regex_v = RegexValidator(r"^\d{3}-\d{4}$")
result = regex_v.validate("123-4567")
```

## Examples

See the `examples/` directory:

- `product_reviews.json` - E-commerce reviews
- `customer_support.yaml` - Support tickets
- `python_functions.json` - Code generation training data

## Workflow

```
1. User Request
   ↓
2. Claude creates dataset_schema.json
   ↓
3. Generate preview (5 rows)
   ↓
4. User feedback → Iterate if needed
   ↓
5. Generate full dataset in batches
   ↓
6. Merge and export to final format
   ↓
7. Deliver to /mnt/user-data/outputs/
```

## Output Formats

- CSV (default)
- JSON
- JSONL
- Parquet

## Comparison with NVIDIA DataDesigner

| Feature | NVIDIA DataDesigner | This Skill |
|---------|---------------------|------------|
| API Keys | Required (OpenAI/NVIDIA) | None needed |
| Setup | pip install + config | Copy to skills |
| LLM Provider | External APIs | Claude native |
| Orchestration | Python framework | Claude + scripts |
| Scale | 1000s of rows | ~100-500 per session |
| Persistence | Files on disk | Session-based |

## Dependencies

- Python 3.9+
- numpy
- scipy
- faker
- pyyaml
- jinja2
- pandas (for Parquet)
- pyarrow (for Parquet)

## License

MIT License
