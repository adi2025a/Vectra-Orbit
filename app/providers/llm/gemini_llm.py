import json
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.interfaces.llm_interface import BaseLLM, LLMChunk, FunctionCall
from app.config import settings

class GeminiLLM(BaseLLM):
    """
    Google Gemini Dialogue Engine using the OpenAI-compatible REST Endpoint.
    Supports low-latency streaming and tool/function calls for call transfers.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    async def generate_response_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        if not self.api_key:
            yield LLMChunk(
                content="[Gemini Error] GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file or environment variables."
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
            try:
                async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield LLMChunk(content=f"[Gemini Error {response.status_code}: {error_text.decode('utf-8')}]")
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

                            # Content chunk
                            content_piece = delta.get("content")
                            if content_piece:
                                yield LLMChunk(content=content_piece)

                            # Function/Tool call chunk
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

            except Exception as e:
                yield LLMChunk(content=f"[Gemini Connection Error: {str(e)}]")
