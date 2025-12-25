# Schema Reference

Complete reference for dataset schema configuration.

## Full Schema Structure

```json
{
  "name": "dataset_name",
  "description": "Optional description",
  "seed": 42,
  "columns": [...],
  "output": {
    "format": "csv",
    "filename": "output",
    "directory": "/mnt/user-data/outputs"
  }
}
```

## Sampler Column Parameters

### category

```json
{
  "name": "product_type",
  "type": "category",
  "params": {
    "values": ["Electronics", "Clothing", "Books"],
    "weights": [0.4, 0.35, 0.25]
  }
}
```

### subcategory

```json
{
  "name": "subcategory",
  "type": "subcategory",
  "params": {
    "category": "product_type",
    "mapping": {
      "Electronics": ["Phones", "Laptops", "Tablets"],
      "Clothing": ["Shirts", "Pants", "Shoes"],
      "Books": ["Fiction", "Non-Fiction", "Technical"]
    }
  }
}
```

### uniform

```json
{
  "name": "rating",
  "type": "uniform",
  "params": {
    "low": 1,
    "high": 5,
    "dtype": "int",
    "decimals": 2
  }
}
```

### gaussian

```json
{
  "name": "price",
  "type": "gaussian",
  "params": {
    "mean": 50.0,
    "std": 15.0,
    "min_val": 0,
    "max_val": 200,
    "dtype": "float",
    "decimals": 2
  }
}
```

### bernoulli

```json
{
  "name": "is_verified",
  "type": "bernoulli",
  "params": {
    "p": 0.7,
    "true_value": true,
    "false_value": false
  }
}
```

### poisson

```json
{
  "name": "event_count",
  "type": "poisson",
  "params": {
    "mean": 5.0
  }
}
```

### datetime

```json
{
  "name": "created_at",
  "type": "datetime",
  "params": {
    "start": "2024-01-01",
    "end": "2024-12-31",
    "format": "%Y-%m-%d %H:%M:%S"
  }
}
```

### person

```json
{
  "name": "customer",
  "type": "person",
  "params": {
    "fields": ["first_name", "last_name", "age", "city", "email", "phone", "company", "job"],
    "age_range": [18, 65],
    "locale": "en_US"
  }
}
```

### uuid

```json
{
  "name": "id",
  "type": "uuid",
  "params": {
    "prefix": "ID-",
    "format": "standard"
  }
}
```

## LLM Column Parameters

### llm_text

```json
{
  "name": "review",
  "type": "llm_text",
  "prompt": "Write a {{ rating }}-star review for {{ product_name }}.",
  "system_prompt": "You are a helpful customer writing reviews.",
  "depends_on": ["rating", "product_name"]
}
```

### llm_code

```json
{
  "name": "code",
  "type": "llm_code",
  "code_lang": "python",
  "prompt": "Write a Python function that {{ description }}.",
  "depends_on": ["description"],
  "validators": [{"type": "python"}]
}
```

### llm_structured

```json
{
  "name": "analysis",
  "type": "llm_structured",
  "prompt": "Analyze: {{ text }}",
  "schema": {
    "type": "object",
    "properties": {
      "sentiment": {"enum": ["positive", "neutral", "negative"]},
      "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    }
  },
  "depends_on": ["text"]
}
```

### llm_judge

```json
{
  "name": "quality_score",
  "type": "llm_judge",
  "prompt": "Rate the quality of: {{ content }}",
  "depends_on": ["content"]
}
```

## Validators

```json
{
  "validators": [
    {"type": "regex", "params": {"pattern": "^[A-Z].*"}},
    {"type": "length", "params": {"min": 10, "max": 500}},
    {"type": "python"},
    {"type": "json_schema", "params": {"schema": {...}}}
  ]
}
```

## Complete Example

```json
{
  "name": "product_reviews",
  "description": "E-commerce product reviews",
  "seed": 42,
  "columns": [
    {
      "name": "review_id",
      "type": "uuid",
      "params": {"prefix": "REV-"}
    },
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
        "fields": ["first_name", "city"],
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
    "filename": "reviews"
  }
}
```
