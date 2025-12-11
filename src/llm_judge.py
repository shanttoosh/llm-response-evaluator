"""
LLM Client wrapper for LLM-as-a-Judge evaluations.

Provides async interface to OpenAI API with retry logic,
caching, and structured output parsing.
"""

import asyncio
import json
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0


class LLMClient:
    """
    Async LLM client with caching and retry logic.
    
    Supports OpenAI API and compatible endpoints.
    Designed for high-throughput evaluation scenarios.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        enable_cache: bool = True
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for evaluations
            base_url: Optional custom API base URL
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            enable_cache: Whether to cache responses
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_cache = enable_cache
        
        # Simple in-memory cache
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
    
    def _get_cache_key(self, prompt: str, system_prompt: str = "") -> str:
        """Generate cache key from prompt content."""
        content = f"{system_prompt}|{prompt}|{self.model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature (0 for deterministic)
            
        Returns:
            LLMResponse with content and metadata
        """
        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, system_prompt)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1
        
        # Rate limiting
        await self._rate_limit()
        
        # Make API call with retries
        response = await self._call_api_with_retry(prompt, system_prompt, temperature)
        
        # Cache the response
        if self.enable_cache:
            self._cache[cache_key] = response
        
        # Track usage
        self.total_requests += 1
        self.total_input_tokens += response.usage.get('prompt_tokens', 0)
        self.total_output_tokens += response.usage.get('completion_tokens', 0)
        
        return response
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that responds in valid JSON format."
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: User prompt expecting JSON response
            system_prompt: System instruction
            
        Returns:
            Parsed JSON dictionary
        """
        response = await self.generate(prompt, system_prompt, temperature=0.0)
        
        try:
            # Try to parse JSON from response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            # Return a default structure on parse failure
            return {
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_content": response.content[:500]
            }
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        now = time.time()
        time_since_last = now - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    async def _call_api_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float
    ) -> LLMResponse:
        """Call API with exponential backoff retry."""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._call_api(prompt, system_prompt, temperature)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    await asyncio.sleep(wait_time)
        
        # Return error response if all retries failed
        return LLMResponse(
            content=json.dumps({"error": str(last_error)}),
            model=self.model,
            usage={},
            latency_seconds=0.0
        )
    
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float
    ) -> LLMResponse:
        """Make actual API call to OpenAI."""
        
        start_time = time.time()
        
        # Try to use openai library if available
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url != "https://api.openai.com/v1" else None
            )
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"} if "json" in system_prompt.lower() else None
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                latency_seconds=latency
            )
            
        except ImportError:
            # Fallback to httpx if openai not installed
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=data["model"],
                usage=data.get("usage", {}),
                latency_seconds=latency
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "cached_items": len(self._cache)
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    
    Returns deterministic responses based on query analysis.
    """
    
    def __init__(self):
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0
    ) -> LLMResponse:
        """Generate mock response."""
        self.total_requests += 1
        
        # Estimate tokens
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = 100
        
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += output_tokens
        
        return LLMResponse(
            content="Mock response for testing",
            model="mock-model",
            usage={"prompt_tokens": int(input_tokens), "completion_tokens": output_tokens},
            latency_seconds=0.1
        )
    
    async def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Generate mock JSON response."""
        self.total_requests += 1
        
        # Return default evaluation scores
        if "relevance" in prompt.lower():
            return {
                "relevance_score": 75,
                "completeness_score": 70,
                "query_understanding_score": 80,
                "addressed_aspects": ["main query addressed"],
                "missing_aspects": [],
                "relevance_issues": [],
                "overall_assessment": "Mock evaluation - reasonable response"
            }
        elif "hallucination" in prompt.lower():
            return {
                "hallucination_score": 80,
                "factual_accuracy_score": 75,
                "supported_claims": ["claim 1", "claim 2"],
                "unsupported_claims": [],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "critical_errors": [],
                "confidence": 70,
                "explanation": "Mock hallucination check"
            }
        else:
            return {"mock": True, "score": 75}
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
