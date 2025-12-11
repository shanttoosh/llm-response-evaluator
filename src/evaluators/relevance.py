"""
Relevance and Completeness Evaluator.

Uses LLM-as-a-Judge to assess how well AI responses address
user queries and whether they provide complete information.
"""

import asyncio
from typing import Optional, Dict, Any, List
from .base import BaseEvaluator, EvaluationResult, EvaluationCategory


# Relevance evaluation prompt template
RELEVANCE_PROMPT = """You are an expert evaluator assessing the relevance and completeness of an AI assistant's response.

USER QUERY:
{user_query}

AI RESPONSE:
{ai_response}

CONTEXT PROVIDED (if any):
{context}

Evaluate the response on these criteria:

1. **RELEVANCE** (0-100): Does the response directly address what the user asked?
   - 90-100: Perfectly addresses the query with precise information
   - 70-89: Addresses the main query with minor tangents
   - 50-69: Partially addresses the query, missing key aspects
   - 0-49: Mostly irrelevant or off-topic

2. **COMPLETENESS** (0-100): Does the response provide all necessary information?
   - 90-100: Comprehensive, covers all aspects the user would need
   - 70-89: Covers main points, minor details missing
   - 50-69: Covers some aspects, significant gaps
   - 0-49: Incomplete, missing critical information

3. **QUERY_UNDERSTANDING** (0-100): Did the AI correctly understand the user's intent?
   - 90-100: Perfect understanding of intent and nuance
   - 70-89: Good understanding with minor misinterpretations
   - 50-69: Partial understanding, some confusion
   - 0-49: Misunderstood the query significantly

Respond in this exact JSON format:
{{
    "relevance_score": <0-100>,
    "completeness_score": <0-100>,
    "query_understanding_score": <0-100>,
    "addressed_aspects": ["list of aspects the response addressed"],
    "missing_aspects": ["list of aspects that should have been addressed but weren't"],
    "relevance_issues": ["specific issues with relevance"],
    "overall_assessment": "brief overall assessment"
}}"""


class RelevanceEvaluator(BaseEvaluator):
    """
    Evaluator for response relevance and completeness.
    
    Uses LLM-as-a-Judge methodology to assess how well
    AI responses address user queries.
    """
    
    def __init__(self, llm_client=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relevance evaluator.
        
        Args:
            llm_client: Optional LLM client for evaluation calls
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.llm_client = llm_client
    
    @property
    def category(self) -> EvaluationCategory:
        return EvaluationCategory.RELEVANCE
    
    @property
    def name(self) -> str:
        return "Relevance & Completeness Evaluator"
    
    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the relevance and completeness of an AI response.
        
        Args:
            user_query: The user's input query
            ai_response: The AI's response to evaluate
            context: Optional context information
            
        Returns:
            EvaluationResult with relevance scores
        """
        if not self.validate_inputs(user_query, ai_response):
            return self.create_error_result("Invalid inputs: empty query or response")
        
        # Use LLM-as-a-Judge if client is available
        if self.llm_client:
            return await self._evaluate_with_llm(user_query, ai_response, context)
        
        # Fallback to heuristic evaluation
        return await self._evaluate_heuristic(user_query, ai_response, context)
    
    async def _evaluate_with_llm(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str]
    ) -> EvaluationResult:
        """Evaluate using LLM-as-a-Judge."""
        try:
            prompt = RELEVANCE_PROMPT.format(
                user_query=user_query,
                ai_response=ai_response,
                context=context or "No specific context provided"
            )
            
            result = await self.llm_client.generate_json(prompt)
            
            # Calculate composite score
            relevance = result.get("relevance_score", 50) / 100
            completeness = result.get("completeness_score", 50) / 100
            understanding = result.get("query_understanding_score", 50) / 100
            
            # Weighted average
            composite_score = (
                relevance * 0.4 +
                completeness * 0.35 +
                understanding * 0.25
            )
            
            issues = result.get("relevance_issues", [])
            missing = result.get("missing_aspects", [])
            if missing:
                issues.extend([f"Missing: {m}" for m in missing])
            
            return EvaluationResult(
                category=self.category,
                score=composite_score,
                confidence=0.85,
                details={
                    "relevance_score": relevance,
                    "completeness_score": completeness,
                    "query_understanding_score": understanding,
                    "addressed_aspects": result.get("addressed_aspects", []),
                    "missing_aspects": missing,
                    "overall_assessment": result.get("overall_assessment", "")
                },
                issues=issues,
                suggestions=self._generate_suggestions(relevance, completeness, missing)
            )
            
        except Exception as e:
            return self.create_error_result(f"LLM evaluation failed: {str(e)}")
    
    async def _evaluate_heuristic(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str]
    ) -> EvaluationResult:
        """
        Fallback heuristic evaluation when LLM is not available.
        
        Uses keyword matching, length analysis, and basic NLP.
        """
        query_words = set(user_query.lower().split())
        response_words = set(ai_response.lower().split())
        
        # Remove common stop words
        stop_words = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or',
                     'and', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'out',
                     'about', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'once', 'here',
                     'there', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own',
                     'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'what'}
        
        query_keywords = query_words - stop_words
        response_keywords = response_words - stop_words
        
        # Calculate keyword overlap (relevance indicator)
        if query_keywords:
            keyword_overlap = len(query_keywords & response_keywords) / len(query_keywords)
        else:
            keyword_overlap = 0.5
        
        # Response length analysis (completeness indicator)
        response_length = len(ai_response.split())
        if response_length < 10:
            length_score = 0.3
        elif response_length < 30:
            length_score = 0.6
        elif response_length < 100:
            length_score = 0.8
        else:
            length_score = 0.9
        
        # Check for question words in query and answer patterns in response
        question_indicators = {'what', 'where', 'when', 'why', 'how', 'who', 'which', 'is', 'are', 'can', 'could', 'would', 'should', 'do', 'does'}
        is_question = bool(query_keywords & question_indicators)
        
        # Check if response seems to answer (has declarative content)
        answer_indicators = {'is', 'are', 'was', 'costs', 'located', 'available', 'offers', 'provides', 'includes'}
        has_answer_pattern = bool(response_keywords & answer_indicators)
        
        if is_question and has_answer_pattern:
            answer_bonus = 0.1
        else:
            answer_bonus = 0.0
        
        # Context utilization (if context provided)
        context_score = 0.5
        if context:
            context_words = set(context.lower().split()) - stop_words
            if context_words:
                context_overlap = len(response_keywords & context_words) / len(context_words)
                context_score = min(1.0, 0.5 + context_overlap)
        
        # Composite score
        relevance_score = min(1.0, keyword_overlap * 0.6 + answer_bonus + 0.2)
        completeness_score = (length_score * 0.5 + context_score * 0.5)
        
        composite_score = (relevance_score * 0.5 + completeness_score * 0.5)
        
        issues = []
        if keyword_overlap < 0.3:
            issues.append("Low keyword overlap with user query")
        if response_length < 20:
            issues.append("Response may be too brief")
        
        return EvaluationResult(
            category=self.category,
            score=composite_score,
            confidence=0.6,  # Lower confidence for heuristic
            details={
                "relevance_score": relevance_score,
                "completeness_score": completeness_score,
                "keyword_overlap": keyword_overlap,
                "response_length": response_length,
                "method": "heuristic"
            },
            issues=issues,
            suggestions=self._generate_suggestions(
                relevance_score, completeness_score, []
            ),
            metadata={"evaluation_method": "heuristic_fallback"}
        )
    
    def _generate_suggestions(
        self,
        relevance: float,
        completeness: float,
        missing_aspects: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []
        
        if relevance < 0.7:
            suggestions.append("Focus more directly on answering the user's specific question")
        
        if completeness < 0.7:
            suggestions.append("Provide more comprehensive information")
        
        if missing_aspects:
            suggestions.append(f"Address these missing aspects: {', '.join(missing_aspects[:3])}")
        
        return suggestions
