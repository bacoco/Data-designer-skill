#!/usr/bin/env python3
"""
Statistical samplers for synthetic data generation.

Inspired by NVIDIA NeMo DataDesigner but simplified for Claude Code skill usage.
Uses scipy.stats for statistical distributions and Faker for persona generation.
"""

import random
import uuid as uuid_module
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
from scipy import stats

# Optional Faker import
try:
    from faker import Faker
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    Faker = None


class SamplerRegistry:
    """Registry for sampler types."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a sampler class."""
        def decorator(sampler_class: type) -> type:
            cls._registry[name] = sampler_class
            return sampler_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a sampler class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown sampler: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, params: dict, seed: int | None = None) -> "BaseSampler":
        """Create a sampler instance."""
        sampler_class = cls.get(name)
        return sampler_class(seed=seed, **params)


class BaseSampler(ABC):
    """Abstract base class for all samplers."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, n: int) -> list[Any]:
        """Generate n samples."""
        pass

    def sample_one(self) -> Any:
        """Generate a single sample."""
        return self.sample(1)[0]


@SamplerRegistry.register("category")
@dataclass
class CategorySampler(BaseSampler):
    """
    Sample from categorical values with optional weights.

    Params:
        values: List of category values
        weights: Optional probability weights (will be normalized)
    """

    values: list[str] = field(default_factory=list)
    weights: list[float] | None = None
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        if self.weights is not None:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def sample(self, n: int) -> list[str]:
        return list(self.rng.choice(self.values, size=n, p=self.weights))


@SamplerRegistry.register("subcategory")
@dataclass
class SubcategorySampler(BaseSampler):
    """
    Sample subcategories based on parent category.

    Params:
        mapping: Dict mapping parent categories to their subcategories
        weights: Optional weights per parent category
    """

    mapping: dict[str, list[str]] = field(default_factory=dict)
    weights: dict[str, list[float]] | None = None
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)

    def sample(self, n: int) -> list[str]:
        """Sample from all subcategories uniformly."""
        all_subcats = [sub for subs in self.mapping.values() for sub in subs]
        return list(self.rng.choice(all_subcats, size=n))

    def sample_for_parents(self, parents: list[str]) -> list[str]:
        """Sample subcategories matching given parent categories."""
        results = []
        for parent in parents:
            if parent not in self.mapping:
                raise ValueError(f"Unknown parent category: {parent}")
            subcats = self.mapping[parent]
            w = None
            if self.weights and parent in self.weights:
                w = self.weights[parent]
                total = sum(w)
                w = [x / total for x in w]
            results.append(self.rng.choice(subcats, p=w))
        return results


@SamplerRegistry.register("uniform")
@dataclass
class UniformSampler(BaseSampler):
    """
    Sample from uniform distribution.

    Params:
        low: Lower bound
        high: Upper bound
        dtype: "int" or "float"
        decimals: Decimal places for float (default 2)
    """

    low: float = 0.0
    high: float = 1.0
    dtype: str = "float"
    decimals: int = 2
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        self._dist = stats.uniform(loc=self.low, scale=self.high - self.low)

    def sample(self, n: int) -> list[int | float]:
        samples = self._dist.rvs(size=n, random_state=self.rng)
        if self.dtype == "int":
            return [int(round(x)) for x in samples]
        return [round(x, self.decimals) for x in samples]


@SamplerRegistry.register("gaussian")
@dataclass
class GaussianSampler(BaseSampler):
    """
    Sample from normal (Gaussian) distribution.

    Params:
        mean: Mean of distribution
        std: Standard deviation
        min_val: Optional minimum clamp
        max_val: Optional maximum clamp
        dtype: "int" or "float"
        decimals: Decimal places for float
    """

    mean: float = 0.0
    std: float = 1.0
    min_val: float | None = None
    max_val: float | None = None
    dtype: str = "float"
    decimals: int = 2
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        self._dist = stats.norm(loc=self.mean, scale=self.std)

    def sample(self, n: int) -> list[float | int]:
        samples = self._dist.rvs(size=n, random_state=self.rng)
        results = []
        for x in samples:
            if self.min_val is not None:
                x = max(x, self.min_val)
            if self.max_val is not None:
                x = min(x, self.max_val)
            if self.dtype == "int":
                results.append(int(round(x)))
            else:
                results.append(round(x, self.decimals))
        return results


@SamplerRegistry.register("bernoulli")
@dataclass
class BernoulliSampler(BaseSampler):
    """
    Sample binary values with probability p.

    Params:
        p: Probability of true_value
        true_value: Value when True (default True)
        false_value: Value when False (default False)
    """

    p: float = 0.5
    true_value: Any = True
    false_value: Any = False
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        self._dist = stats.bernoulli(p=self.p)

    def sample(self, n: int) -> list[Any]:
        bits = self._dist.rvs(size=n, random_state=self.rng)
        return [self.true_value if b else self.false_value for b in bits]


@SamplerRegistry.register("poisson")
@dataclass
class PoissonSampler(BaseSampler):
    """
    Sample from Poisson distribution.

    Params:
        mean: Mean (lambda) of distribution
    """

    mean: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        self._dist = stats.poisson(mu=self.mean)

    def sample(self, n: int) -> list[int]:
        return list(self._dist.rvs(size=n, random_state=self.rng))


@SamplerRegistry.register("datetime")
@dataclass
class DateTimeSampler(BaseSampler):
    """
    Sample random datetimes within a range.

    Params:
        start: Start date (string or datetime)
        end: End date (string or datetime)
        format: Output format string (default "%Y-%m-%d %H:%M:%S")
    """

    start: str | datetime = "2024-01-01"
    end: str | datetime = "2024-12-31"
    format: str = "%Y-%m-%d %H:%M:%S"
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        # Parse dates
        if isinstance(self.start, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    self.start = datetime.strptime(self.start, fmt)
                    break
                except ValueError:
                    continue
        if isinstance(self.end, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    self.end = datetime.strptime(self.end, fmt)
                    break
                except ValueError:
                    continue

        self._delta_seconds = (self.end - self.start).total_seconds()

    def sample(self, n: int) -> list[str]:
        offsets = self.rng.uniform(0, self._delta_seconds, size=n)
        results = []
        for off in offsets:
            dt = self.start + timedelta(seconds=off)
            results.append(dt.strftime(self.format))
        return results


@SamplerRegistry.register("uuid")
@dataclass
class UUIDSampler(BaseSampler):
    """
    Generate unique identifiers.

    Params:
        prefix: Optional prefix string
        version: UUID version (4 is random)
        format: "standard", "hex", or "urn"
    """

    prefix: str = ""
    version: int = 4
    format: str = "standard"
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)

    def sample(self, n: int) -> list[str]:
        results = []
        for _ in range(n):
            uid = uuid_module.uuid4()
            if self.format == "hex":
                val = uid.hex
            elif self.format == "urn":
                val = uid.urn
            else:
                val = str(uid)
            results.append(f"{self.prefix}{val}")
        return results


@SamplerRegistry.register("person")
@dataclass
class PersonSampler(BaseSampler):
    """
    Generate synthetic persona data using Faker.

    Params:
        fields: List of fields to generate
            Supported: first_name, last_name, name, age, email, city,
                      state, country, address, phone, company, job, username
        age_range: Tuple of (min_age, max_age)
        locale: Faker locale (default "en_US")
    """

    fields: list[str] = field(default_factory=lambda: ["first_name", "last_name", "age", "city", "email"])
    age_range: tuple[int, int] = (18, 70)
    locale: str = "en_US"
    seed: int | None = None

    def __post_init__(self):
        super().__init__(self.seed)
        if HAS_FAKER:
            Faker.seed(self.seed)
            self._faker = Faker(self.locale)
        else:
            self._faker = None

    def sample(self, n: int) -> list[dict[str, Any]]:
        if not HAS_FAKER:
            # Fallback without Faker
            return self._sample_fallback(n)

        results = []
        for _ in range(n):
            person = {}
            for f in self.fields:
                if f == "first_name":
                    person[f] = self._faker.first_name()
                elif f == "last_name":
                    person[f] = self._faker.last_name()
                elif f == "name":
                    person[f] = self._faker.name()
                elif f == "age":
                    person[f] = int(self.rng.integers(self.age_range[0], self.age_range[1] + 1))
                elif f == "email":
                    person[f] = self._faker.email()
                elif f == "city":
                    person[f] = self._faker.city()
                elif f == "state":
                    person[f] = self._faker.state()
                elif f == "country":
                    person[f] = self._faker.country()
                elif f == "address":
                    person[f] = self._faker.address().replace("\n", ", ")
                elif f == "phone":
                    person[f] = self._faker.phone_number()
                elif f == "company":
                    person[f] = self._faker.company()
                elif f == "job":
                    person[f] = self._faker.job()
                elif f == "username":
                    person[f] = self._faker.user_name()
                elif hasattr(self._faker, f):
                    person[f] = getattr(self._faker, f)()
                else:
                    person[f] = None
            results.append(person)
        return results

    def _sample_fallback(self, n: int) -> list[dict[str, Any]]:
        """Fallback person generation without Faker."""
        first_names = ["John", "Jane", "Alex", "Sam", "Chris", "Pat", "Jordan", "Taylor"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle"]

        results = []
        for _ in range(n):
            person = {}
            for f in self.fields:
                if f == "first_name":
                    person[f] = self.rng.choice(first_names)
                elif f == "last_name":
                    person[f] = self.rng.choice(last_names)
                elif f == "name":
                    person[f] = f"{self.rng.choice(first_names)} {self.rng.choice(last_names)}"
                elif f == "age":
                    person[f] = int(self.rng.integers(self.age_range[0], self.age_range[1] + 1))
                elif f == "email":
                    name = self.rng.choice(first_names).lower()
                    person[f] = f"{name}{self.rng.integers(100, 999)}@example.com"
                elif f == "city":
                    person[f] = self.rng.choice(cities)
                else:
                    person[f] = None
            results.append(person)
        return results


def generate_samples(column_config: dict, n: int, seed: int | None = None) -> list[Any]:
    """
    Generate samples for a single column configuration.

    Args:
        column_config: Column configuration with 'type' and 'params'
        n: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated samples
    """
    col_type = column_config.get("type")
    params = column_config.get("params", {})

    sampler = SamplerRegistry.create(col_type, params, seed=seed)
    return sampler.sample(n)


if __name__ == "__main__":
    # Example usage
    import json

    print("Testing samplers...")

    # Category sampler
    cat = CategorySampler(values=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=42)
    print(f"Category: {cat.sample(5)}")

    # Uniform sampler
    uni = UniformSampler(low=1, high=5, dtype="int", seed=42)
    print(f"Uniform int: {uni.sample(5)}")

    # Gaussian sampler
    gauss = GaussianSampler(mean=50, std=10, min_val=0, max_val=100, seed=42)
    print(f"Gaussian: {gauss.sample(5)}")

    # DateTime sampler
    dt = DateTimeSampler(start="2024-01-01", end="2024-12-31", format="%Y-%m-%d", seed=42)
    print(f"DateTime: {dt.sample(3)}")

    # Person sampler
    person = PersonSampler(fields=["first_name", "age", "city"], seed=42)
    print(f"Person: {json.dumps(person.sample(2), indent=2)}")

    # UUID sampler
    uid = UUIDSampler(prefix="ID-", seed=42)
    print(f"UUID: {uid.sample(2)}")

    print("\nAll samplers working!")
