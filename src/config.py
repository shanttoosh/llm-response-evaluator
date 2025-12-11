"""
Configuration module for LLM Evaluation Pipeline.

Handles environment variables, API settings, and evaluation thresholds.
Designed for easy customization and scalability at production scale.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM provider settings."""
    
    provider: str = "openai"
    model: str = "gpt-4o-mini"  # Cost-effective model for evaluations
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    
    # API call settings
    max_retries: int = 3
    timeout_seconds: int = 30
    temperature: float = 0.0  # Deterministic outputs for evaluation consistency
    
    # Rate limiting for scale
    max_concurrent_requests: int = 10
    requests_per_minute: int = 60


@dataclass
class EvaluationConfig:
    """Configuration for evaluation thresholds and weights."""
    
    # Relevance thresholds
    relevance_weight: float = 0.3
    completeness_weight: float = 0.2
    
    # Hallucination detection thresholds
    hallucination_weight: float = 0.3
    factual_accuracy_weight: float = 0.2
    
    # Score thresholds
    high_quality_threshold: float = 0.8
    acceptable_threshold: float = 0.6
    
    # Context matching settings
    min_context_similarity: float = 0.7


@dataclass
class CostConfig:
    """Token pricing configuration for cost calculations."""
    
    # GPT-4o-mini pricing (per 1M tokens)
    input_cost_per_million: float = 0.15  # $0.15 per 1M input tokens
    output_cost_per_million: float = 0.60  # $0.60 per 1M output tokens
    
    # Alternative model pricing
    gpt4_input_cost_per_million: float = 2.50
    gpt4_output_cost_per_million: float = 10.00


@dataclass
class CacheConfig:
    """Configuration for caching strategies to minimize latency and costs."""
    
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 10000  # Maximum cached items
    
    # Content hashing for cache keys
    use_content_hash: bool = True


@dataclass
class PipelineConfig:
    """Main configuration aggregating all settings."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Batch processing settings
    batch_size: int = 10
    enable_async: bool = True
    
    # Output settings
    verbose: bool = True
    output_format: str = "json"  # json, console, or both


# Global configuration instance
config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> PipelineConfig:
    """Update configuration with provided values."""
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.llm, key):
            setattr(config.llm, key, value)
        elif hasattr(config.evaluation, key):
            setattr(config.evaluation, key, value)
    
    return config
