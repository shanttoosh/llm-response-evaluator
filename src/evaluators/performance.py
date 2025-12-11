"""
Performance Evaluator for Latency and Cost Tracking.

Measures response latency and calculates token-based costs
for LLM API usage monitoring.
"""

import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from .base import BaseEvaluator, EvaluationResult, EvaluationCategory


class PerformanceEvaluator(BaseEvaluator):
    """
    Evaluator for latency and cost metrics.
    
    Tracks response generation time and estimates API costs
    based on token counts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance evaluator.
        
        Args:
            config: Optional configuration with cost parameters
        """
        super().__init__(config)
        
        # Default cost configuration (GPT-4o-mini pricing per 1M tokens)
        self.input_cost_per_million = config.get('input_cost_per_million', 0.15) if config else 0.15
        self.output_cost_per_million = config.get('output_cost_per_million', 0.60) if config else 0.60
        
        # Latency thresholds (in seconds)
        self.excellent_latency = 2.0
        self.good_latency = 5.0
        self.acceptable_latency = 10.0
    
    @property
    def category(self) -> EvaluationCategory:
        return EvaluationCategory.LATENCY
    
    @property
    def name(self) -> str:
        return "Latency & Cost Evaluator"
    
    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate latency and cost metrics.
        
        Args:
            user_query: The user's input query
            ai_response: The AI's response to evaluate
            context: Optional context information
            **kwargs: Additional parameters:
                - latency_seconds: Actual latency if measured
                - user_timestamp: Timestamp of user message
                - ai_timestamp: Timestamp of AI response
                - input_tokens: Actual input token count
                - output_tokens: Actual output token count
            
        Returns:
            EvaluationResult with latency and cost metrics
        """
        if not self.validate_inputs(user_query, ai_response):
            return self.create_error_result("Invalid inputs: empty query or response")
        
        # Extract timestamps if provided
        latency_seconds = kwargs.get('latency_seconds')
        user_timestamp = kwargs.get('user_timestamp')
        ai_timestamp = kwargs.get('ai_timestamp')
        
        # Calculate latency from timestamps if not directly provided
        if latency_seconds is None and user_timestamp and ai_timestamp:
            latency_seconds = self._calculate_latency(user_timestamp, ai_timestamp)
        
        # Estimate tokens
        input_tokens = kwargs.get('input_tokens') or self._estimate_tokens(user_query)
        output_tokens = kwargs.get('output_tokens') or self._estimate_tokens(ai_response)
        context_tokens = self._estimate_tokens(context) if context else 0
        
        total_input_tokens = input_tokens + context_tokens
        
        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        total_cost = input_cost + output_cost
        
        # Calculate latency score
        latency_score = self._calculate_latency_score(latency_seconds)
        
        # Calculate cost efficiency score (based on output value per dollar)
        cost_score = self._calculate_cost_score(total_cost, output_tokens)
        
        # Composite score (latency weighted more for real-time)
        if latency_seconds is not None:
            composite_score = latency_score * 0.6 + cost_score * 0.4
        else:
            composite_score = cost_score  # Only cost if no latency data
        
        issues = []
        if latency_seconds and latency_seconds > self.acceptable_latency:
            issues.append(f"High latency: {latency_seconds:.2f}s exceeds {self.acceptable_latency}s threshold")
        
        if total_cost > 0.01:  # More than 1 cent per response
            issues.append(f"High cost per response: ${total_cost:.4f}")
        
        return EvaluationResult(
            category=self.category,
            score=composite_score,
            confidence=0.95,  # High confidence for deterministic metrics
            details={
                "latency_seconds": latency_seconds,
                "latency_score": latency_score,
                "input_tokens": total_input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_input_tokens + output_tokens,
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_1k_tokens": round(total_cost / ((total_input_tokens + output_tokens) / 1000), 6) if (total_input_tokens + output_tokens) > 0 else 0,
                "tokens_per_second": round(output_tokens / latency_seconds, 2) if latency_seconds and latency_seconds > 0 else None
            },
            issues=issues,
            suggestions=self._generate_suggestions(latency_seconds, total_cost, output_tokens)
        )
    
    def _calculate_latency(self, user_ts: str, ai_ts: str) -> Optional[float]:
        """Calculate latency between timestamps."""
        try:
            # Parse ISO format timestamps
            user_dt = datetime.fromisoformat(user_ts.replace("Z", "+00:00"))
            ai_dt = datetime.fromisoformat(ai_ts.replace("Z", "+00:00"))
            return (ai_dt - user_dt).total_seconds()
        except (ValueError, AttributeError):
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses a simple heuristic: ~4 characters per token for English.
        For production, use tiktoken for accurate counts.
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 chars per token for English
        # This is a simplification; tiktoken would be more accurate
        char_count = len(text)
        word_count = len(text.split())
        
        # Average between character-based and word-based estimates
        char_based = char_count / 4
        word_based = word_count * 1.3
        
        return int((char_based + word_based) / 2)
    
    def _calculate_latency_score(self, latency: Optional[float]) -> float:
        """Calculate normalized latency score (higher is better)."""
        if latency is None:
            return 0.5  # Neutral score when no data
        
        if latency <= self.excellent_latency:
            return 1.0
        elif latency <= self.good_latency:
            # Linear interpolation between excellent and good
            return 1.0 - 0.2 * (latency - self.excellent_latency) / (self.good_latency - self.excellent_latency)
        elif latency <= self.acceptable_latency:
            # Linear interpolation between good and acceptable
            return 0.8 - 0.3 * (latency - self.good_latency) / (self.acceptable_latency - self.good_latency)
        else:
            # Exponential decay for very slow responses
            return max(0.1, 0.5 * (self.acceptable_latency / latency))
    
    def _calculate_cost_score(self, cost: float, output_tokens: int) -> float:
        """Calculate cost efficiency score."""
        if cost <= 0:
            return 1.0
        
        # Score based on cost per meaningful output
        # Lower cost per token = higher score
        cost_per_token = cost / max(output_tokens, 1)
        
        # Thresholds in dollars per token
        excellent_threshold = 0.000001  # $0.001 per 1000 tokens
        poor_threshold = 0.00001  # $0.01 per 1000 tokens
        
        if cost_per_token <= excellent_threshold:
            return 1.0
        elif cost_per_token >= poor_threshold:
            return 0.3
        else:
            # Linear interpolation
            ratio = (cost_per_token - excellent_threshold) / (poor_threshold - excellent_threshold)
            return 1.0 - 0.7 * ratio
    
    def _generate_suggestions(
        self,
        latency: Optional[float],
        cost: float,
        output_tokens: int
    ) -> List[str]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        if latency and latency > self.acceptable_latency:
            suggestions.append("Consider using a faster model or reducing context size")
            suggestions.append("Implement response streaming for better perceived latency")
        
        if cost > 0.001:  # More than $0.001 per response
            suggestions.append("Consider using a smaller model for routine queries")
            suggestions.append("Implement context pruning to reduce input tokens")
        
        if output_tokens > 500:
            suggestions.append("Consider truncating or summarizing long responses")
        
        return suggestions


class CostCalculator:
    """
    Utility class for aggregating costs across multiple evaluations.
    
    Useful for batch processing and reporting.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with model-specific pricing."""
        self.model = model
        self._pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
    
    def add_request(self, input_tokens: int, output_tokens: int):
        """Add a request to the cumulative totals."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        pricing = self._pricing.get(self.model, self._pricing["gpt-4o-mini"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        self.total_cost += input_cost + output_cost
        self.request_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "model": self.model,
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "average_cost_per_request": round(self.total_cost / max(self.request_count, 1), 6)
        }
