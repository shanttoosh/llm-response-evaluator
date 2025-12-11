"""
Main Evaluation Pipeline Orchestrator.

Coordinates all evaluators to produce comprehensive evaluation
reports for AI chat responses.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .data_loader import (
    ChatConversation, ContextVectors,
    load_chat_conversation, load_context_vectors
)
from .evaluators import (
    RelevanceEvaluator, HallucinationEvaluator, PerformanceEvaluator,
    TurnEvaluation
)
from .llm_judge import LLMClient, MockLLMClient
from .config import get_config, PipelineConfig


@dataclass
class ConversationEvaluation:
    """Complete evaluation results for a conversation."""
    
    chat_id: int
    user_id: int
    evaluation_timestamp: str
    turns_evaluated: int
    turn_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "turns_evaluated": self.turns_evaluated,
            "summary": self.summary,
            "turn_evaluations": self.turn_evaluations,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class EvaluationPipeline:
    """
    Main orchestrator for evaluating AI chat responses.
    
    Coordinates relevance, hallucination, and performance evaluators
    to produce comprehensive quality assessments.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        llm_client: Optional[LLMClient] = None,
        use_mock_llm: bool = False
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Pipeline configuration
            llm_client: Optional pre-configured LLM client
            use_mock_llm: Whether to use mock LLM for testing
        """
        self.config = config or get_config()
        
        # Initialize LLM client
        if use_mock_llm:
            self.llm_client = MockLLMClient()
        elif llm_client:
            self.llm_client = llm_client
        elif self.config.llm.api_key:
            self.llm_client = LLMClient(
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                max_retries=self.config.llm.max_retries,
                timeout=self.config.llm.timeout_seconds,
                enable_cache=self.config.cache.enable_caching
            )
        else:
            # No API key - use mock or heuristic only
            self.llm_client = None
        
        # Initialize evaluators
        self.relevance_evaluator = RelevanceEvaluator(
            llm_client=self.llm_client,
            config={"weights": self.config.evaluation}
        )
        self.hallucination_evaluator = HallucinationEvaluator(
            llm_client=self.llm_client,
            config={"weights": self.config.evaluation}
        )
        self.performance_evaluator = PerformanceEvaluator(
            config={
                "input_cost_per_million": self.config.cost.input_cost_per_million,
                "output_cost_per_million": self.config.cost.output_cost_per_million
            }
        )
    
    async def evaluate_turn(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str] = None,
        user_timestamp: Optional[str] = None,
        ai_timestamp: Optional[str] = None
    ) -> TurnEvaluation:
        """
        Evaluate a single conversation turn.
        
        Args:
            user_query: The user's message
            ai_response: The AI's response
            context: Grounding context from vector database
            user_timestamp: Timestamp of user message
            ai_timestamp: Timestamp of AI response
            
        Returns:
            TurnEvaluation with all metric results
        """
        results = []
        
        # Run all evaluators concurrently for efficiency
        if self.config.enable_async:
            eval_tasks = [
                self.relevance_evaluator.evaluate(user_query, ai_response, context),
                self.hallucination_evaluator.evaluate(user_query, ai_response, context),
                self.performance_evaluator.evaluate(
                    user_query, ai_response, context,
                    user_timestamp=user_timestamp,
                    ai_timestamp=ai_timestamp
                )
            ]
            results = await asyncio.gather(*eval_tasks)
        else:
            # Sequential evaluation
            rel_result = await self.relevance_evaluator.evaluate(
                user_query, ai_response, context
            )
            results.append(rel_result)
            hal_result = await self.hallucination_evaluator.evaluate(
                user_query, ai_response, context
            )
            results.append(hal_result)
            perf_result = await self.performance_evaluator.evaluate(
                user_query, ai_response, context,
                user_timestamp=user_timestamp,
                ai_timestamp=ai_timestamp
            )
            results.append(perf_result)
        
        return TurnEvaluation(
            turn_number=0,  # Will be set by caller
            user_query=user_query,
            ai_response=ai_response,
            results=results
        )
    
    async def evaluate_conversation(
        self,
        conversation: ChatConversation,
        context: ContextVectors
    ) -> ConversationEvaluation:
        """
        Evaluate all AI responses in a conversation.
        
        Args:
            conversation: The chat conversation to evaluate
            context: Context vectors for grounding checks
            
        Returns:
            ConversationEvaluation with all turn results
        """
        # Get context text for grounding
        context_text = context.get_context_text()
        
        # Get turn pairs (user query -> AI response)
        turn_pairs = conversation.get_turn_pairs()
        
        turn_evaluations = []
        all_scores = {
            "relevance": [],
            "hallucination": [],
            "performance": []
        }
        
        # Evaluate each AI response
        for user_turn, ai_turn in turn_pairs:
            # Evaluate this turn
            turn_eval = await self.evaluate_turn(
                user_query=user_turn.message,
                ai_response=ai_turn.message,
                context=context_text,
                user_timestamp=user_turn.created_at,
                ai_timestamp=ai_turn.created_at
            )
            turn_eval.turn_number = ai_turn.turn
            
            # Collect scores by category
            for result in turn_eval.results:
                category = result.category.value
                if category in all_scores:
                    all_scores[category].append(result.score)
            
            # Check for known evaluation notes (ground truth)
            if ai_turn.evaluation_note:
                turn_eval_dict = turn_eval.to_dict()
                turn_eval_dict["ground_truth_note"] = ai_turn.evaluation_note
                turn_evaluations.append(turn_eval_dict)
            else:
                turn_evaluations.append(turn_eval.to_dict())
        
        # Calculate summary statistics
        summary = self._calculate_summary(all_scores, turn_evaluations)
        
        # Add cost tracking metadata
        metadata = {
            "context_tokens_total": context.get_total_tokens(),
            "context_vectors_count": len(context.vector_data),
            "vectors_used_count": (
                len(context.sources.vectors_used) if context.sources else 0
            ),
            "evaluation_mode": (
                "llm_judge" if self.llm_client
                and not isinstance(self.llm_client, MockLLMClient)
                else "heuristic"
            )
        }
        
        if hasattr(self.llm_client, 'get_usage_stats'):
            metadata["llm_usage"] = self.llm_client.get_usage_stats()
        
        return ConversationEvaluation(
            chat_id=conversation.chat_id,
            user_id=conversation.user_id,
            evaluation_timestamp=datetime.utcnow().isoformat() + "Z",
            turns_evaluated=len(turn_evaluations),
            turn_evaluations=turn_evaluations,
            summary=summary,
            metadata=metadata
        )
    
    def _calculate_summary(
        self,
        all_scores: Dict[str, List[float]],
        turn_evaluations: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the evaluation."""
        summary = {
            "overall_score": 0.0,
            "category_scores": {},
            "issues_count": 0,
            "quality_distribution": {
                "excellent": 0,
                "good": 0,
                "acceptable": 0,
                "poor": 0
            }
        }
        
        # Calculate average scores per category
        for category, scores in all_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                summary["category_scores"][category] = round(avg, 4)
        
        # Calculate overall score (weighted average)
        weights = {
            "relevance": 0.35,
            "hallucination": 0.40,
            "performance": 0.25
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, avg_score in summary["category_scores"].items():
            weight = weights.get(category, 0.25)
            weighted_sum += avg_score * weight
            total_weight += weight
        
        if total_weight > 0:
            summary["overall_score"] = round(weighted_sum / total_weight, 4)
        
        # Count issues and quality distribution
        for turn_eval in turn_evaluations:
            overall = turn_eval.get("overall_score", 0.5)
            
            if overall >= 0.9:
                summary["quality_distribution"]["excellent"] += 1
            elif overall >= 0.7:
                summary["quality_distribution"]["good"] += 1
            elif overall >= 0.5:
                summary["quality_distribution"]["acceptable"] += 1
            else:
                summary["quality_distribution"]["poor"] += 1
            
            # Count issues
            for eval_result in turn_eval.get("evaluations", []):
                summary["issues_count"] += len(eval_result.get("issues", []))
        
        return summary
    
    async def evaluate_from_files(
        self,
        conversation_path: str,
        context_path: str
    ) -> ConversationEvaluation:
        """
        Evaluate from file paths (convenience method).
        
        Args:
            conversation_path: Path to conversation JSON
            context_path: Path to context vectors JSON
            
        Returns:
            ConversationEvaluation results
        """
        conversation = load_chat_conversation(conversation_path)
        context = load_context_vectors(context_path)
        
        return await self.evaluate_conversation(conversation, context)


async def run_evaluation(
    conversation_path: str,
    context_path: str,
    use_mock_llm: bool = True,
    verbose: bool = True
) -> ConversationEvaluation:
    """
    Run a complete evaluation on conversation and context files.
    
    This is the main entry point for running evaluations.
    
    Args:
        conversation_path: Path to conversation JSON file
        context_path: Path to context vectors JSON file
        use_mock_llm: Whether to use mock LLM (True for testing without API)
        verbose: Whether to print progress information
        
    Returns:
        ConversationEvaluation with complete results
    """
    if verbose:
        print(f"Loading conversation from: {conversation_path}")
        print(f"Loading context from: {context_path}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(use_mock_llm=use_mock_llm)
    
    # Run evaluation
    if verbose:
        print("Running evaluation...")
    
    result = await pipeline.evaluate_from_files(conversation_path, context_path)
    
    if verbose:
        print("\nEvaluation complete!")
        print(f"Turns evaluated: {result.turns_evaluated}")
        print(f"Overall score: {result.summary.get('overall_score', 0):.2%}")
        print(f"Total issues found: {result.summary.get('issues_count', 0)}")
    
    return result
