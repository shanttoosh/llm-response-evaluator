"""
Hallucination and Factual Accuracy Evaluator.

Detects unsupported claims and factual inaccuracies by
cross-referencing AI responses against provided context.
"""

import asyncio
import re
from typing import Optional, Dict, Any, List, Tuple
from .base import BaseEvaluator, EvaluationResult, EvaluationCategory


# Hallucination detection prompt template
HALLUCINATION_PROMPT = """You are an expert fact-checker evaluating an AI response for hallucinations and factual accuracy.

IMPORTANT: A hallucination is when the AI makes claims that are:
1. NOT supported by the provided context
2. Contradicted by the provided context
3. Fabricated information presented as fact

USER QUERY:
{user_query}

AI RESPONSE:
{ai_response}

SOURCE CONTEXT (Ground Truth):
{context}

Carefully analyze the AI response and identify:

1. **SUPPORTED CLAIMS**: Statements that ARE backed by the context
2. **UNSUPPORTED CLAIMS**: Statements NOT found in the context (potential hallucinations)
3. **CONTRADICTED CLAIMS**: Statements that CONTRADICT the context (definite hallucinations)
4. **UNVERIFIABLE CLAIMS**: Statements that cannot be verified from context but aren't necessarily wrong

Respond in this exact JSON format:
{{
    "hallucination_score": <0-100 where 100 means NO hallucinations>,
    "factual_accuracy_score": <0-100 where 100 means perfectly accurate>,
    "supported_claims": ["list of claims that are supported by context"],
    "unsupported_claims": ["list of claims NOT found in context"],
    "contradicted_claims": ["list of claims that CONTRADICT context"],
    "unverifiable_claims": ["list of claims that can't be verified"],
    "critical_errors": ["any critical factual errors that could cause harm"],
    "confidence": <0-100 confidence in this evaluation>,
    "explanation": "brief explanation of findings"
}}"""


class HallucinationEvaluator(BaseEvaluator):
    """
    Evaluator for detecting hallucinations and checking factual accuracy.
    
    Cross-references AI responses against provided context to identify
    unsupported, contradicted, or fabricated claims.
    """
    
    def __init__(self, llm_client=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hallucination evaluator.
        
        Args:
            llm_client: Optional LLM client for evaluation calls
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.llm_client = llm_client
    
    @property
    def category(self) -> EvaluationCategory:
        return EvaluationCategory.HALLUCINATION
    
    @property
    def name(self) -> str:
        return "Hallucination & Factual Accuracy Evaluator"
    
    async def evaluate(
        self,
        user_query: str,
        ai_response: str,
        context: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the AI response for hallucinations.
        
        Args:
            user_query: The user's input query
            ai_response: The AI's response to evaluate
            context: Context information for grounding (required for hallucination detection)
            
        Returns:
            EvaluationResult with hallucination detection results
        """
        if not self.validate_inputs(user_query, ai_response):
            return self.create_error_result("Invalid inputs: empty query or response")
        
        if not context:
            # Without context, we can only do limited analysis
            return await self._evaluate_without_context(user_query, ai_response)
        
        # Use LLM-as-a-Judge if client is available
        if self.llm_client:
            return await self._evaluate_with_llm(user_query, ai_response, context)
        
        # Fallback to heuristic evaluation
        return await self._evaluate_heuristic(user_query, ai_response, context)
    
    async def _evaluate_with_llm(
        self,
        user_query: str,
        ai_response: str,
        context: str
    ) -> EvaluationResult:
        """Evaluate using LLM-as-a-Judge."""
        try:
            # Truncate context if too long (keep most relevant parts)
            max_context_len = 4000
            if len(context) > max_context_len:
                context = context[:max_context_len] + "\n... [truncated]"
            
            prompt = HALLUCINATION_PROMPT.format(
                user_query=user_query,
                ai_response=ai_response,
                context=context
            )
            
            result = await self.llm_client.generate_json(prompt)
            
            # Calculate scores
            hallucination_score = result.get("hallucination_score", 50) / 100
            factual_score = result.get("factual_accuracy_score", 50) / 100
            confidence = result.get("confidence", 70) / 100
            
            # Composite score (weighted towards hallucination detection)
            composite_score = (hallucination_score * 0.6 + factual_score * 0.4)
            
            # Collect issues
            issues = []
            unsupported = result.get("unsupported_claims", [])
            contradicted = result.get("contradicted_claims", [])
            critical = result.get("critical_errors", [])
            
            for claim in contradicted:
                issues.append(f"CONTRADICTION: {claim}")
            for claim in unsupported[:3]:  # Limit to top 3
                issues.append(f"UNSUPPORTED: {claim}")
            for error in critical:
                issues.append(f"CRITICAL ERROR: {error}")
            
            return EvaluationResult(
                category=self.category,
                score=composite_score,
                confidence=confidence,
                details={
                    "hallucination_score": hallucination_score,
                    "factual_accuracy_score": factual_score,
                    "supported_claims_count": len(result.get("supported_claims", [])),
                    "unsupported_claims_count": len(unsupported),
                    "contradicted_claims_count": len(contradicted),
                    "supported_claims": result.get("supported_claims", []),
                    "unsupported_claims": unsupported,
                    "contradicted_claims": contradicted,
                    "unverifiable_claims": result.get("unverifiable_claims", []),
                    "explanation": result.get("explanation", "")
                },
                issues=issues,
                suggestions=self._generate_suggestions(
                    hallucination_score, unsupported, contradicted
                )
            )
            
        except Exception as e:
            return self.create_error_result(f"LLM evaluation failed: {str(e)}")
    
    async def _evaluate_without_context(
        self,
        user_query: str,
        ai_response: str
    ) -> EvaluationResult:
        """Limited evaluation when no context is provided."""
        # Without context, we can only check for obvious red flags
        
        issues = []
        red_flags = 0
        
        # Check for overly specific unverifiable claims
        specific_patterns = [
            r'\$[\d,]+',  # Specific dollar amounts
            r'Rs\.?\s*[\d,]+',  # Specific rupee amounts
            r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)',  # Specific times
            r'\d+%',  # Specific percentages
            r'\b\d{3,}\s*(rupees|dollars|patients|cases)\b',  # Specific numbers
        ]
        
        for pattern in specific_patterns:
            matches = re.findall(pattern, ai_response)
            if matches:
                red_flags += len(matches) * 0.1
        
        # Check for absolute statements
        absolute_words = ['always', 'never', 'definitely', 'certainly', 'guaranteed', 'proven']
        for word in absolute_words:
            if word.lower() in ai_response.lower():
                red_flags += 0.1
                issues.append(f"Uses absolute language: '{word}'")
        
        # Score is lower without context (less confident)
        base_score = max(0.3, 1.0 - red_flags)
        
        return EvaluationResult(
            category=self.category,
            score=base_score,
            confidence=0.3,  # Very low confidence without context
            details={
                "red_flags_detected": red_flags,
                "specific_claims_found": red_flags > 0,
                "method": "no_context_analysis"
            },
            issues=issues if issues else ["No context provided for thorough hallucination check"],
            suggestions=["Provide context for more accurate hallucination detection"],
            metadata={"evaluation_method": "limited_no_context"}
        )
    
    async def _evaluate_heuristic(
        self,
        user_query: str,
        ai_response: str,
        context: str
    ) -> EvaluationResult:
        """
        Fallback heuristic evaluation when LLM is not available.
        
        Uses text matching and overlap analysis.
        """
        # Extract potential facts from response
        facts = self._extract_facts(ai_response)
        
        # Check each fact against context
        supported_facts = []
        unsupported_facts = []
        
        context_lower = context.lower()
        
        for fact in facts:
            # Check if key terms from fact appear in context
            fact_terms = set(fact.lower().split()) - self._get_stop_words()
            
            if len(fact_terms) == 0:
                continue
            
            matches = sum(1 for term in fact_terms if term in context_lower)
            match_ratio = matches / len(fact_terms) if fact_terms else 0
            
            if match_ratio >= 0.5:
                supported_facts.append(fact)
            else:
                unsupported_facts.append(fact)
        
        # Calculate scores
        total_facts = len(supported_facts) + len(unsupported_facts)
        if total_facts > 0:
            support_ratio = len(supported_facts) / total_facts
        else:
            support_ratio = 0.7  # Default when no clear facts detected
        
        # Check for specific numbers/prices that should be in context
        numbers_in_response = re.findall(r'Rs\.?\s*[\d,]+|\$[\d,]+|[\d]+(?:\.\d+)?%', ai_response)
        numbers_in_context = re.findall(r'Rs\.?\s*[\d,]+|\$[\d,]+|[\d]+(?:\.\d+)?%', context)
        
        unmatched_numbers = []
        for num in numbers_in_response:
            # Normalize and check
            normalized = re.sub(r'[^\d.]', '', num)
            if not any(normalized in re.sub(r'[^\d.]', '', ctx_num) for ctx_num in numbers_in_context):
                unmatched_numbers.append(num)
        
        # Penalize for unverified specific numbers
        number_penalty = min(0.3, len(unmatched_numbers) * 0.1)
        
        composite_score = max(0.0, support_ratio - number_penalty)
        
        issues = []
        if unsupported_facts:
            issues.append(f"{len(unsupported_facts)} potentially unsupported claims detected")
        if unmatched_numbers:
            issues.append(f"Unverified specific values: {', '.join(unmatched_numbers[:3])}")
        
        return EvaluationResult(
            category=self.category,
            score=composite_score,
            confidence=0.5,  # Moderate confidence for heuristic
            details={
                "supported_facts_count": len(supported_facts),
                "unsupported_facts_count": len(unsupported_facts),
                "unmatched_numbers": unmatched_numbers,
                "support_ratio": support_ratio,
                "method": "heuristic"
            },
            issues=issues,
            suggestions=self._generate_suggestions(
                composite_score,
                unsupported_facts,
                []
            ),
            metadata={"evaluation_method": "heuristic_fallback"}
        )
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract potential factual claims from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Look for declarative statements with facts
            fact_indicators = [
                r'costs?\s+(?:Rs\.?|₹|\$)\s*[\d,]+',
                r'(?:is|are|was|were)\s+(?:located|available|open)',
                r'\d+\s*(?:minutes?|hours?|days?|weeks?)',
                r'offers?\s+',
                r'provides?\s+',
                r'includes?\s+',
            ]
            
            for pattern in fact_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    facts.append(sentence)
                    break
        
        return facts
    
    def _get_stop_words(self) -> set:
        """Get common stop words for filtering."""
        return {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or',
                'and', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'out',
                'this', 'that', 'these', 'those', 'you', 'your', 'we', 'our'}
    
    def _generate_suggestions(
        self,
        hallucination_score: float,
        unsupported: List[str],
        contradicted: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on findings."""
        suggestions = []
        
        if contradicted:
            suggestions.append("Review and correct statements that contradict source material")
        
        if unsupported and len(unsupported) > 2:
            suggestions.append("Ensure all factual claims are grounded in provided context")
        
        if hallucination_score < 0.7:
            suggestions.append("Consider adding disclaimers for information not in source material")
        
        return suggestions
