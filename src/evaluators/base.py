"""
Base evaluator class and common evaluation result structures.

Provides abstract base class for all evaluators to ensure
consistent interface and result formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class EvaluationCategory(Enum):
    """Categories of evaluation metrics."""
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    HALLUCINATION = "hallucination"
    FACTUAL_ACCURACY = "factual_accuracy"
    LATENCY = "latency"
    COST = "cost"


class QualityLevel(Enum):
    """Quality classification for evaluation results."""
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"           # >= 0.7
    ACCEPTABLE = "acceptable"  # >= 0.5
    POOR = "poor"           # < 0.5


@dataclass
class EvaluationResult:
    """
    Standardized evaluation result from any evaluator.
    
    Attributes:
        category: The evaluation category (relevance, hallucination, etc.)
        score: Normalized score from 0.0 to 1.0
        confidence: Confidence in the evaluation (0.0 to 1.0)
        details: Detailed breakdown of the evaluation
        issues: List of identified issues or concerns
        suggestions: Improvement suggestions
        metadata: Additional evaluation metadata
    """
    
    category: EvaluationCategory
    score: float  # 0.0 to 1.0, higher is better
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality_level(self) -> QualityLevel:
        """Classify the quality based on score."""
        if self.score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.score >= 0.7:
            return QualityLevel.GOOD
        elif self.score >= 0.5:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    @property
    def passed(self) -> bool:
        """Check if evaluation meets acceptable threshold."""
        return self.score >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "quality_level": self.quality_level.value,
            "passed": self.passed,
            "details": self.details,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class TurnEvaluation:
    """Evaluation results for a single conversation turn."""
    
    turn_number: int
    user_query: str
    ai_response: str
    results: List[EvaluationResult] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        if not self.results:
            return 0.0
        
        # Weight by confidence
        total_weight = sum(r.confidence for r in self.results)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(r.score * r.confidence for r in self.results)
        return weighted_sum / total_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_number": self.turn_number,
            "user_query": self.user_query[:100] + "..." if len(self.user_query) > 100 else self.user_query,
            "ai_response": self.ai_response[:200] + "..." if len(self.ai_response) > 200 else self.ai_response,
            "overall_score": round(self.overall_score, 4),
            "evaluations": [r.to_dict() for r in self.results]
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Subclasses must implement the evaluate method to perform
    specific evaluation logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    @property
    @abstractmethod
    def category(self) -> EvaluationCategory:
        """Return the evaluation category this evaluator handles."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of the evaluator."""
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the AI response.
        
        Args:
            user_query: The user's input query
            ai_response: The AI's response to evaluate
            context: Optional context information for grounding
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with score and details
        """
        pass
    
    def validate_inputs(
        self,
        user_query: str,
        ai_response: str
    ) -> bool:
        """
        Validate input data before evaluation.
        
        Args:
            user_query: The user's input query
            ai_response: The AI's response
            
        Returns:
            True if inputs are valid
        """
        if not user_query or not user_query.strip():
            return False
        if not ai_response or not ai_response.strip():
            return False
        return True
    
    def create_error_result(
        self,
        error_message: str
    ) -> EvaluationResult:
        """
        Create an error result when evaluation fails.
        
        Args:
            error_message: Description of the error
            
        Returns:
            EvaluationResult with zero score and error details
        """
        return EvaluationResult(
            category=self.category,
            score=0.0,
            confidence=0.0,
            details={"error": error_message},
            issues=[f"Evaluation failed: {error_message}"]
        )
