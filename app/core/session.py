import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.interfaces.vad_interface import BaseVAD
from app.interfaces.stt_interface import BaseSTT
from app.interfaces.llm_interface import BaseLLM, LLMChunk, FunctionCall
from app.interfaces.tts_interface import BaseTTS
from app.interfaces.telephony_interface import BaseTelephonyAdapter
from app.core.metrics import TurnMetric, LatencyTracker, CallMetricsSummary
from app.core.tool_registry import get_default_tools
from app.db.repository import CallRepository

class CallSession:
    """
    State machine and Orchestrator for an active AI Voice Call Session.
    Manages audio buffering, VAD triggers, cascaded pipeline execution, 
    barge-in cancellation, tool execution (call transfer), and latency metrics.
    """
    def __init__(
        self,
        session_id: str,
        telephony_adapter: BaseTelephonyAdapter,
        vad: BaseVAD,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        db_session: Optional[AsyncSession] = None,
        system_prompt: Optional[str] = None,
        campaign_id: Optional[str] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.telephony = telephony_adapter
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.db = db_session

        self.system_prompt = system_prompt or (
            "You are a friendly, professional AI calling agent. Your objective is to assist the caller with their campaign or support inquiry. "
            "If the customer asks to speak to a human or requires customer support handoff, use the 'transfer_call' tool."
        )
        self.campaign_id = campaign_id

        # Conversation History
        self.messages: List[Dict[str, Any]] = []
        self.turn_counter: int = 0
        self.db_call_id: Optional[str] = None

        # Audio Buffers & State
        self.audio_buffer = bytearray()
        self.is_user_speaking: bool = False
        self.silence_chunks: int = 0
        self.speech_chunks: int = 0

        # Interruption / Barge-in Task Management
        self.active_response_task: Optional[asyncio.Task] = None
        self.metrics_summary = CallMetricsSummary(session_id=self.session_id)

    async def initialize(self):
        """Create DB record for the call session."""
        if self.db:
            db_call = await CallRepository.create_call(
                session=self.db,
                call_sid=self.session_id,
                telephony_provider=type(self.telephony).__name__,
                campaign_id=self.campaign_id
            )
            self.db_call_id = db_call.id

    async def process_incoming_audio(self, pcm_bytes: bytes):
        """Process incoming 20ms audio frame from client."""
        if not pcm_bytes:
            return

        is_speech = self.vad.process_chunk(pcm_bytes)

        if is_speech:
            self.speech_chunks += 1
            self.silence_chunks = 0
            self.audio_buffer.extend(pcm_bytes)

            if not self.is_user_speaking and self.speech_chunks >= 3:
                # User started speaking!
                self.is_user_speaking = True
                # Trigger Barge-in Interruption if AI is currently synthesizing or talking
                await self.interrupt_ai_speech()

        else:
            self.silence_chunks += 1
            if self.is_user_speaking:
                self.audio_buffer.extend(pcm_bytes)

            # Silence threshold reached after speech -> User finished speaking turn
            if self.is_user_speaking and self.silence_chunks >= 12:  # ~240ms silence
                self.is_user_speaking = False
                self.speech_chunks = 0
                self.silence_chunks = 0
                
                # Clone audio buffer for processing turn
                speech_data = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                self.vad.reset()

                # Trigger Cascaded Pipeline in async task
                self.active_response_task = asyncio.create_task(self.run_cascaded_pipeline(speech_data))

    async def interrupt_ai_speech(self):
        """Cancel ongoing TTS/LLM task when user interrupts (Barge-in)."""
        if self.active_response_task and not self.active_response_task.done():
            self.active_response_task.cancel()
            self.active_response_task = None
            print(f"[Session {self.session_id}] ⚡ BARGE-IN DETECTED: Cancelled active AI response.")

    async def run_cascaded_pipeline(self, audio_data: bytes):
        """
        Executes the cascaded voice pipeline:
        VAD -> STT -> LLM -> TTS -> Telephony Out
        Recording granular latencies for every module.
        """
        self.turn_counter += 1
        turn = TurnMetric(turn_id=self.turn_counter)
        tracker = LatencyTracker()

        user_speech_stop_time = time.perf_counter()

        try:
            # 1. STT Module Latency
            tracker.start("stt")
            user_transcript = await self.stt.transcribe(audio_data)
            turn.stt_latency_ms = tracker.stop("stt")

            if not user_transcript.strip():
                return

            print(f"[Session {self.session_id}] User (STT {turn.stt_latency_ms:.1f}ms): {user_transcript}")

            # Append user message & DB turn
            self.messages.append({"role": "user", "content": user_transcript})
            if self.db and self.db_call_id:
                await CallRepository.add_transcript_turn(
                    self.db, self.db_call_id, self.turn_counter, "user", user_transcript
                )

            # 2. LLM Dialogue Engine Latency (TTFT & Total)
            tracker.start("llm_ttft")
            tracker.start("llm_total")
            
            async def text_stream_generator():
                first_token = True
                llm_response_text = ""
                
                async for chunk in self.llm.generate_response_stream(
                    messages=self.messages,
                    tools=get_default_tools(),
                    system_prompt=self.system_prompt
                ):
                    if first_token and chunk.content:
                        turn.llm_ttft_ms = tracker.stop("llm_ttft")
                        first_token = False
                    
                    if chunk.content:
                        llm_response_text += chunk.content
                        yield chunk.content

                    if chunk.function_call:
                        await self.handle_function_call(chunk.function_call)

                turn.llm_total_latency_ms = tracker.stop("llm_total")
                if llm_response_text:
                    self.messages.append({"role": "assistant", "content": llm_response_text})
                    if self.db and self.db_call_id:
                        await CallRepository.add_transcript_turn(
                            self.db, self.db_call_id, self.turn_counter, "assistant", llm_response_text
                        )

            # 3. TTS Synthesis Stream Latency
            tracker.start("tts_first_chunk")
            tracker.start("tts_total")
            first_audio = True

            async for audio_chunk in self.tts.synthesize_stream(text_stream_generator()):
                if first_audio:
                    turn.tts_first_chunk_ms = tracker.stop("tts_first_chunk")
                    # Calculate E2E Voice Latency (User stopped speaking -> 1st audio frame sent)
                    turn.e2e_voice_latency_ms = (time.perf_counter() - user_speech_stop_time) * 1000.0
                    first_audio = False

                # Stream audio to telephony / browser client
                await self.telephony.send_audio_chunk(self.session_id, audio_chunk)

            turn.tts_total_latency_ms = tracker.stop("tts_total")

            # 4. Latency Analysis & Bottleneck Identification
            turn.calculate_bottleneck()
            self.metrics_summary.turns.append(turn)
            self.metrics_summary.aggregate()

            print(
                f"[Session {self.session_id}] Turn {self.turn_counter} Latency breakdown: "
                f"STT={turn.stt_latency_ms:.1f}ms | LLM TTFT={turn.llm_ttft_ms:.1f}ms | "
                f"TTS 1st={turn.tts_first_chunk_ms:.1f}ms | E2E={turn.e2e_voice_latency_ms:.1f}ms | "
                f"🔥 BOTTLENECK: {turn.max_latency_module}"
            )

            # Save metrics to DB
            if self.db and self.db_call_id:
                await CallRepository.record_turn_metric(self.db, self.db_call_id, turn)

        except asyncio.CancelledError:
            print(f"[Session {self.session_id}] Pipeline execution cancelled due to barge-in.")
        except Exception as e:
            print(f"[Session {self.session_id}] Error in voice pipeline: {e}")

    async def handle_function_call(self, fn: FunctionCall):
        """Handle AI Dialogue Engine function calls (e.g. transfer_call handoff)."""
        print(f"[Session {self.session_id}] 🛠️ Tool Call Triggered: {fn.name}({fn.arguments})")
        if fn.name == "transfer_call":
            target_num = fn.arguments.get("target_phone_number", "+18005550199")
            reason = fn.arguments.get("reason", "Customer support handoff")
            
            # Execute call forwarding via Telephony Adapter
            transferred = await self.telephony.transfer_call(self.session_id, target_num)
            
            if self.db and self.db_call_id:
                await CallRepository.mark_call_completed(
                    self.db,
                    self.db_call_id,
                    status="transferred",
                    was_transferred=True,
                    transfer_target=target_num,
                    transfer_reason=reason
                )

    async def close(self):
        """End call session and persist metrics summary."""
        if self.active_response_task and not self.active_response_task.done():
            self.active_response_task.cancel()
        
        if self.db and self.db_call_id:
            await CallRepository.mark_call_completed(self.db, self.db_call_id, status="completed")
