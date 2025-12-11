"""
Data Loader module for parsing input JSON files.

Handles chat conversations and context vectors with strong typing
and validation using Pydantic models.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    turn: int
    sender_id: int
    role: str  # "User" or "AI/Chatbot"
    message: str
    created_at: str
    evaluation_note: Optional[str] = None
    
    @property
    def is_ai_response(self) -> bool:
        """Check if this turn is from the AI."""
        return self.role == "AI/Chatbot"
    
    @property
    def is_user_message(self) -> bool:
        """Check if this turn is from the user."""
        return self.role == "User"
    
    @property
    def timestamp(self) -> Optional[datetime]:
        """Parse the timestamp if available."""
        try:
            return datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None


@dataclass
class ChatConversation:
    """Represents a complete chat conversation."""
    
    chat_id: int
    user_id: int
    conversation_turns: List[ConversationTurn]
    
    def get_ai_responses(self) -> List[ConversationTurn]:
        """Get all AI response turns."""
        return [t for t in self.conversation_turns if t.is_ai_response]
    
    def get_user_messages(self) -> List[ConversationTurn]:
        """Get all user message turns."""
        return [t for t in self.conversation_turns if t.is_user_message]
    
    def get_turn_pairs(self) -> List[tuple]:
        """Get user query and AI response pairs."""
        pairs = []
        for i, turn in enumerate(self.conversation_turns):
            if turn.is_ai_response and i > 0:
                # Find the preceding user message
                for j in range(i - 1, -1, -1):
                    if self.conversation_turns[j].is_user_message:
                        pairs.append((self.conversation_turns[j], turn))
                        break
        return pairs
    
    def get_response_latency(self, ai_turn: ConversationTurn) -> Optional[float]:
        """Calculate latency between user message and AI response in seconds."""
        turn_idx = self.conversation_turns.index(ai_turn)
        if turn_idx > 0:
            prev_turn = self.conversation_turns[turn_idx - 1]
            if prev_turn.is_user_message:
                ai_ts = ai_turn.timestamp
                user_ts = prev_turn.timestamp
                if ai_ts and user_ts:
                    return (ai_ts - user_ts).total_seconds()
        return None


@dataclass
class VectorData:
    """Represents a single context vector from the vector database."""
    
    id: int
    text: str
    source_url: Optional[str] = None
    tokens: int = 0
    created_at: Optional[str] = None
    score: Optional[float] = None  # Similarity score
    
    @property
    def source_domain(self) -> Optional[str]:
        """Extract domain from source URL."""
        if self.source_url:
            try:
                from urllib.parse import urlparse
                return urlparse(self.source_url).netloc
            except Exception:
                return None
        return None


@dataclass
class VectorInfo:
    """Information about a vector match including score."""
    
    score: float
    vector_id: int
    tokens_count: int


@dataclass
class SourcesData:
    """Sources information from the context vectors response."""
    
    message_id: int
    vector_ids: List[int]
    vectors_info: List[VectorInfo]
    vectors_used: List[int]
    final_response: List[str]
    
    @property
    def response_text(self) -> str:
        """Get the complete final response as a single string."""
        return " ".join(self.final_response)


@dataclass
class ContextVectors:
    """Represents context vectors fetched from vector database."""
    
    status: str
    status_code: int
    message: str
    vector_data: List[VectorData]
    sources: Optional[SourcesData] = None
    
    def get_context_text(self) -> str:
        """Get all context text concatenated."""
        return "\n\n".join([v.text for v in self.vector_data])
    
    def get_used_context(self) -> List[VectorData]:
        """Get only the vectors that were actually used for the response."""
        if self.sources and self.sources.vectors_used:
            used_ids = set(self.sources.vectors_used)
            return [v for v in self.vector_data if v.id in used_ids]
        return self.vector_data[:3]  # Default to top 3 if not specified
    
    def get_total_tokens(self) -> int:
        """Get total token count of all context vectors."""
        return sum(v.tokens for v in self.vector_data)


def _parse_json_with_comments(content: str) -> Dict[str, Any]:
    """
    Parse JSON that may contain JavaScript-style comments.
    
    Handles:
    - Single line comments: // ...
    - Trailing commas before closing brackets
    - Both Windows (CRLF) and Unix (LF) line endings
    
    Args:
        content: JSON string that may contain comments
        
    Returns:
        Parsed JSON as dictionary
    """
    import re
    
    # Use splitlines() which handles all line ending types properly
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        # Skip lines that are pure comments
        if stripped.startswith('//'):
            continue
        cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Remove trailing commas before closing brackets/braces
    # This handles cases where a comment line followed an element with a comma
    cleaned_content = re.sub(r',(\s*[\]\}])', r'\1', cleaned_content)
    
    return json.loads(cleaned_content)


def load_chat_conversation(file_path: str) -> ChatConversation:
    """
    Load a chat conversation from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        ChatConversation object
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat conversation file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse JSON with comments using helper function
    data = _parse_json_with_comments(content)
    
    turns = [
        ConversationTurn(
            turn=t['turn'],
            sender_id=t['sender_id'],
            role=t['role'],
            message=t['message'],
            created_at=t['created_at'],
            evaluation_note=t.get('evaluation_note')
        )
        for t in data['conversation_turns']
    ]
    
    return ChatConversation(
        chat_id=data['chat_id'],
        user_id=data['user_id'],
        conversation_turns=turns
    )


def load_context_vectors(file_path: str) -> ContextVectors:
    """
    Load context vectors from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        ContextVectors object
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Context vectors file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse JSON (handles trailing commas and comments if present)
    data = _parse_json_with_comments(content)
    
    # Parse vector data
    vector_data = []
    for v in data['data']['vector_data']:
        vector_data.append(VectorData(
            id=v['id'],
            text=v.get('text', ''),
            source_url=v.get('source_url'),
            tokens=v.get('tokens', 0),
            created_at=v.get('created_at')
        ))
    
    # Parse sources if available
    sources = None
    if 'sources' in data['data']:
        s = data['data']['sources']
        vectors_info = [
            VectorInfo(
                score=vi['score'],
                vector_id=vi['vector_id'] if isinstance(vi['vector_id'], int) else int(vi['vector_id']),
                tokens_count=vi['tokens_count']
            )
            for vi in s.get('vectors_info', [])
        ]
        sources = SourcesData(
            message_id=s['message_id'],
            vector_ids=[int(vid) if isinstance(vid, str) else vid for vid in s.get('vector_ids', [])],
            vectors_info=vectors_info,
            vectors_used=[int(vid) if isinstance(vid, str) else vid for vid in s.get('vectors_used', [])],
            final_response=s.get('final_response', [])
        )
    
    return ContextVectors(
        status=data['status'],
        status_code=data['status_code'],
        message=data['message'],
        vector_data=vector_data,
        sources=sources
    )


@dataclass
class EvaluationInput:
    """Combined input for evaluation pipeline."""
    
    conversation: ChatConversation
    context: ContextVectors
    
    @classmethod
    def from_files(cls, conversation_path: str, context_path: str) -> 'EvaluationInput':
        """Load evaluation input from file paths."""
        return cls(
            conversation=load_chat_conversation(conversation_path),
            context=load_context_vectors(context_path)
        )
