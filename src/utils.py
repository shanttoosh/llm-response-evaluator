"""
Utility functions for the LLM Evaluation Pipeline.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove empty markdown links
    text = re.sub(r'\[\]\([^)]+\)', '', text)
    
    return text.strip()


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s\)\]<>\"\']+|www\.[^\s\)\]<>\"\']+' 
    return re.findall(url_pattern, text)


def calculate_text_hash(text: str) -> str:
    """Calculate MD5 hash of text for caching."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_score(score: float, as_percentage: bool = True) -> str:
    """Format a score for display."""
    if as_percentage:
        return f"{score * 100:.1f}%"
    return f"{score:.4f}"


def get_quality_label(score: float) -> str:
    """Get quality label for a score."""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Acceptable"
    else:
        return "Poor"


def get_quality_emoji(score: float) -> str:
    """Get emoji indicator for quality level."""
    if score >= 0.9:
        return "✅"
    elif score >= 0.7:
        return "👍"
    elif score >= 0.5:
        return "⚠️"
    else:
        return "❌"


def merge_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple evaluation results into a summary."""
    if not evaluations:
        return {}
    
    merged = {
        "total_evaluations": len(evaluations),
        "scores": {},
        "all_issues": [],
        "all_suggestions": []
    }
    
    for eval_result in evaluations:
        category = eval_result.get("category", "unknown")
        score = eval_result.get("score", 0)
        
        if category not in merged["scores"]:
            merged["scores"][category] = []
        merged["scores"][category].append(score)
        
        merged["all_issues"].extend(eval_result.get("issues", []))
        merged["all_suggestions"].extend(eval_result.get("suggestions", []))
    
    # Calculate averages
    merged["average_scores"] = {
        cat: sum(scores) / len(scores)
        for cat, scores in merged["scores"].items()
    }
    
    # Deduplicate issues and suggestions
    merged["unique_issues"] = list(set(merged["all_issues"]))
    merged["unique_suggestions"] = list(set(merged["all_suggestions"]))
    
    return merged


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses heuristic of ~4 characters per token for English.
    For accurate counts, use tiktoken library.
    """
    if not text:
        return 0
    
    # Count based on characters and words
    char_count = len(text)
    word_count = len(text.split())
    
    # Average of character-based and word-based estimate
    char_estimate = char_count / 4
    word_estimate = word_count * 1.3
    
    return int((char_estimate + word_estimate) / 2)


def format_cost(cost_usd: float) -> str:
    """Format cost in USD for display."""
    if cost_usd < 0.0001:
        return f"${cost_usd:.6f}"
    elif cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"


def parse_iso_timestamp(timestamp: str) -> Optional[str]:
    """Parse and normalize ISO timestamp."""
    if not timestamp:
        return None
    
    # Handle Z suffix
    if timestamp.endswith('Z'):
        timestamp = timestamp[:-1] + '+00:00'
    
    return timestamp
