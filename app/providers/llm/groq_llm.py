import json
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.interfaces.llm_interface import BaseLLM, LLMChunk, FunctionCall
from app.config import settings

class GroqLLM(BaseLLM):
    """
    Groq Cloud LLM Dialogue Engine (Free Cloud Tier).
    Ultra-low latency streaming (<100ms TTFT) with native tool/function calling support.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        if not self.api_key:
            yield LLMChunk(
                content="I'm sorry, the Groq API key is not configured. Please set your GROQ_API_KEY in environment variables or switch to Mock LLM."
            )
            return

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 300,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield LLMChunk(content=f"[LLM Error {response.status_code}: {error_text.decode('utf-8')}]")
                    return

                tool_name = ""
                tool_args_str = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk_json = json.loads(data_str)
                        delta = chunk_json["choices"][0].get("delta", {})

                        # Check content delta
                        content_piece = delta.get("content")
                        if content_piece:
                            yield LLMChunk(content=content_piece)

                        # Check function / tool call delta
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            tc = tool_calls[0]
                            if "function" in tc:
                                fn = tc["function"]
                                if "name" in fn and fn["name"]:
                                    tool_name = fn["name"]
                                if "arguments" in fn and fn["arguments"]:
                                    tool_args_str += fn["arguments"]

                    except Exception:
                        continue

                if tool_name:
                    try:
                        args = json.loads(tool_args_str) if tool_args_str else {}
                    except Exception:
                        args = {}
                    yield LLMChunk(
                        function_call=FunctionCall(name=tool_name, arguments=args),
                        is_final=True
                    )


class OllamaLLM(BaseLLM):
    """Local Ollama LLM provider (Zero-cost local instance)."""
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL

    async def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            msg = data.get("message", {}).get("content", "")
                            if msg:
                                yield LLMChunk(content=msg)
            except Exception as e:
                yield LLMChunk(content=f"[Ollama Error: Make sure Ollama server is running at {self.base_url}. Error: {str(e)}]")


class MockLLM(BaseLLM):
    """High-speed Offline Mock LLM for rapid UI & Latency testing."""
    async def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "").lower()
                break

        if "forward" in user_msg or "transfer" in user_msg or "agent" in user_msg or "customer support" in user_msg:
            # Trigger Call Transfer Tool Call
            yield LLMChunk(content="Certainly! Connecting you to a live customer care representative right now. Please hold on.")
            yield LLMChunk(
                function_call=FunctionCall(
                    name="transfer_call",
                    arguments={"target_phone_number": "+18005550199", "reason": "Customer requested human support agent"}
                ),
                is_final=True
            )
        else:
            response_text = (
                "Hello! Thank you for calling. I am your AI assistant. "
                "I can help you with your marketing campaign or answer any product questions. "
                "If you need further assistance, I can also transfer your call directly to a human representative."
            )
            for word in response_text.split(" "):
                yield LLMChunk(content=word + " ")
