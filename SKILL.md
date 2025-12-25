---
name: Data Designer
description: Generate high-quality synthetic datasets using statistical samplers and Claude's native LLM capabilities
version: 1.0.0
triggers:
  - synthetic data
  - generate dataset
  - create dataset
  - data generation
  - fake data
  - mock data
  - test data
  - training data
---

# Data Designer Skill

Generate synthetic datasets by combining statistical samplers with Claude's LLM capabilities. No external API keys required.

## Capabilities

### Statistical Samplers (No LLM Required)
- **category** - Sample from weighted categorical values
- **subcategory** - Hierarchical sampling based on parent category
- **uniform** - Uniform distribution (int or float)
- **gaussian** - Normal distribution with mean/std
- **bernoulli** - Binary with probability
- **poisson** - Poisson distribution
- **datetime** - Random dates in range
- **person** - Synthetic personas (name, age, email, city, etc.)
- **uuid** - Unique identifiers

### LLM-Generated Columns (Claude Native)
- **llm_text** - Free-form text generation
- **llm_code** - Code with syntax validation
- **llm_structured** - JSON matching schema
- **llm_judge** - Quality scoring of other columns

### Jinja2 Templating
Reference other columns in prompts:
```
Write a {{ rating }}-star review for {{ product_name }} by {{ customer.first_name }}.
```

### Validators
- **regex** - Pattern matching
- **length** - Min/max character count
- **python** - Python syntax validation (AST + ruff)
- **json_schema** - JSON schema validation

## Workflow Protocol

When the user requests synthetic data generation, follow this protocol:

### Step 1: Clarify Requirements
Ask about:
- Dataset purpose (training, testing, demos)
- Number of records needed
- Column names and types
- Any specific distributions or constraints
- Output format (CSV, JSON, JSONL, Parquet)

### Step 2: Generate Schema
Create a `dataset_schema.json` file defining all columns:

```json
{
  "name": "product_reviews",
  "description": "Customer reviews for e-commerce",
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
      "params": { "low": 1, "high": 5, "dtype": "int" }
    },
    {
      "name": "customer",
      "type": "person",
      "params": {
        "fields": ["first_name", "last_name", "age", "city", "email"],
        "age_range": [18, 65]
      }
    },
    {
      "name": "review_text",
      "type": "llm_text",
      "prompt": "Write a {{ rating }}-star review for a {{ product_category }} product by {{ customer.first_name }} from {{ customer.city }}. 2-3 sentences.",
      "depends_on": ["rating", "product_category", "customer"]
    }
  ],
  "output": {
    "format": "csv",
    "filename": "product_reviews"
  }
}
```

### Step 3: Generate Preview Batch
Run the batch generator for a small preview (3-5 rows):

```bash
python scripts/batch_generator.py --schema dataset_schema.json --rows 5 --output preview.json
```

Show the preview to the user and ask for feedback.

### Step 4: Iterate on Schema
If the user requests changes:
1. Update the schema file
2. Regenerate preview
3. Repeat until user approves

### Step 5: Generate Full Dataset
Once approved, generate in batches:

```bash
# Generate batches
python scripts/batch_generator.py --schema dataset_schema.json --rows 100 --batch-size 20 --output batches/

# Merge batches
python scripts/merger.py --input batches/ --output final_dataset.csv
```

### Step 6: Validate (Optional)
Run validators if configured:

```bash
python scripts/validator.py --input final_dataset.csv --schema dataset_schema.json
```

### Step 7: Deliver
- Save to `/mnt/user-data/outputs/` for download
- Show summary statistics
- Offer to generate more or modify

## Column Dependencies

Columns are generated in dependency order (topological sort):
1. Sampler columns first (no dependencies)
2. LLM columns in order of their `depends_on` fields
3. Expression/validation columns last

## LLM Generation Strategy

For LLM columns, Claude generates content directly in the conversation:
1. Render the Jinja2 prompt with row data
2. Generate the content
3. Validate if validators configured
4. Retry on validation failure (max 3 attempts)

For large datasets (>50 rows), generate in batches to manage context.

## File Locations

- Schema: `dataset_schema.json`
- Batches: `batches/batch_001.json`, `batch_002.json`, ...
- Output: `/mnt/user-data/outputs/{filename}.{format}`

## Example Requests

**Simple:**
> "Generate 50 product reviews with ratings 1-5"

**Complex:**
> "Create a dataset of 200 customer support tickets with:
> - Ticket ID (UUID)
> - Customer name and email
> - Category (billing, technical, general)
> - Priority (1-5, gaussian around 3)
> - Description (LLM generated based on category)
> - Resolution (LLM generated if priority > 3)"

**Code Training:**
> "Generate 100 Python function examples with:
> - Function description
> - Python code (validated)
> - Test cases"

## Tips

- Use `seed` in schema for reproducibility
- Preview first, then scale up
- For correlated columns, use subcategory or expression types
- Validate code columns with python validator
- Keep LLM prompts specific and concise for better quality
