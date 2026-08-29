from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Optional, List, Dict, Any

from app.db.models import Call, CallTranscript, CallMetric, CallAnalytics, Campaign
from app.core.metrics import TurnMetric

class CallRepository:
    """Async database operations for Call Sessions, Transcripts, and Latency Metrics."""

    @staticmethod
    async def create_call(
        session: AsyncSession,
        call_sid: str,
        telephony_provider: str = "web",
        campaign_id: Optional[str] = None,
        customer_phone: Optional[str] = None
    ) -> Call:
        call = Call(
            call_sid=call_sid,
            telephony_provider=telephony_provider,
            campaign_id=campaign_id,
            customer_phone=customer_phone,
            status="active",
            start_time=datetime.now(timezone.utc),
        )
        session.add(call)
        await session.commit()
        await session.refresh(call)
        return call

    @staticmethod
    async def add_transcript_turn(
        session: AsyncSession,
        call_id: str,
        turn_index: int,
        role: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CallTranscript:
        transcript = CallTranscript(
            call_id=call_id,
            turn_index=turn_index,
            role=role,
            text=text,
            extra_metadata=metadata or {}
        )
        session.add(transcript)
        await session.commit()
        return transcript

    @staticmethod
    async def record_turn_metric(
        session: AsyncSession,
        call_id: str,
        metric: TurnMetric
    ) -> CallMetric:
        call_metric = CallMetric(
            call_id=call_id,
            turn_index=metric.turn_id,
            vad_latency_ms=metric.vad_latency_ms,
            stt_latency_ms=metric.stt_latency_ms,
            llm_ttft_ms=metric.llm_ttft_ms,
            llm_total_latency_ms=metric.llm_total_latency_ms,
            tts_first_chunk_ms=metric.tts_first_chunk_ms,
            tts_total_latency_ms=metric.tts_total_latency_ms,
            e2e_voice_latency_ms=metric.e2e_voice_latency_ms,
            max_latency_module=metric.max_latency_module
        )
        session.add(call_metric)
        await session.commit()
        return call_metric

    @staticmethod
    async def mark_call_completed(
        session: AsyncSession,
        call_id: str,
        status: str = "completed",
        was_transferred: bool = False,
        transfer_target: Optional[str] = None,
        transfer_reason: Optional[str] = None
    ):
        result = await session.execute(select(Call).where(Call.id == call_id))
        call = result.scalar_one_or_none()
        if call:
            end_dt = datetime.now(timezone.utc)
            call.end_time = end_dt
            if call.start_time:
                start_dt = call.start_time.replace(tzinfo=timezone.utc) if call.start_time.tzinfo is None else call.start_time
                call.duration_seconds = (end_dt - start_dt).total_seconds()
            call.status = status
            call.was_transferred = was_transferred
            call.transfer_target = transfer_target
            call.transfer_reason = transfer_reason
            await session.commit()

    @staticmethod
    async def get_call_details(session: AsyncSession, call_id: str) -> Optional[Call]:
        stmt = (
            select(Call)
            .where(Call.id == call_id)
            .options(
                selectinload(Call.transcripts),
                selectinload(Call.metrics),
                selectinload(Call.analytics)
            )
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_recent_calls(session: AsyncSession, limit: int = 20) -> List[Call]:
        stmt = (
            select(Call)
            .order_by(Call.start_time.desc())
            .limit(limit)
            .options(
                selectinload(Call.metrics),
                selectinload(Call.transcripts)
            )
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
