from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class LLMChunk(BaseModel):
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    is_final: bool = False

class BaseLLM(ABC):
    """Abstract Interface for Dialogue Engine / LLM Provider."""

    @abstractmethod
    async def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        """
        Stream LLM text chunks and potential function calls given a conversation history.
        """
        pass
