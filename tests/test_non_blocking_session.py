import pytest
import pytest_asyncio
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.db.models import Base
from app.db.repository import CallRepository
from app.core.session import CallSession
from app.core.metrics import TurnMetric
from app.providers.factory import ProviderFactory

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

class DummyTelephony:
    def __init__(self):
        self.sent_events = []
        self.sent_audio = []

    async def send_audio_chunk(self, session_id: str, audio_bytes: bytes) -> None:
        self.sent_audio.append(audio_bytes)

    async def send_control_event(self, event_dict: dict) -> None:
        self.sent_events.append(event_dict)

    async def transfer_call(self, session_id: str, target_phone_number: str) -> bool:
        return True

    async def end_call(self, session_id: str, reason: str = "completed") -> None:
        pass

@pytest_asyncio.fixture
async def async_db():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

@pytest.mark.asyncio
async def test_non_blocking_session_queue(async_db):
    telephony = DummyTelephony()
    vad = ProviderFactory.get_vad("energy")
    stt = ProviderFactory.get_stt("mock")
    llm = ProviderFactory.get_llm("mock")
    tts = ProviderFactory.get_tts("mock")

    session = CallSession(
        session_id="test-non-blocking-1",
        telephony_adapter=telephony,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        db_session=async_db
    )

    await session.initialize()
    assert session.db_call_id is not None
    assert session._db_worker_task is not None

    # Simulate non-blocking DB writes (which use _enqueue_db_job without await)
    session._enqueue_db_job(
        CallRepository.add_transcript_turn,
        session.db_call_id,
        1,
        "user",
        "Hello from automated non-blocking test"
    )

    metric = TurnMetric(
        turn_id=1,
        stt_latency_ms=10.0,
        llm_ttft_ms=20.0,
        tts_first_chunk_ms=15.0,
        e2e_voice_latency_ms=45.0
    )
    session._enqueue_db_job(
        CallRepository.record_turn_metric,
        session.db_call_id,
        metric
    )

    # Calling close() must flush and drain the entire queue
    await session.close()

    # Query DB to verify that all queued jobs were cleanly persisted
    call = await CallRepository.get_call_details(async_db, session.db_call_id)
    assert call is not None
    assert call.status == "completed"
    assert len(call.transcripts) == 1
    assert call.transcripts[0].text == "Hello from automated non-blocking test"
    assert len(call.metrics) == 1
    assert call.metrics[0].e2e_voice_latency_ms == 45.0

@pytest.mark.asyncio
async def test_pipeline_run_non_blocking_persistence(async_db):
    telephony = DummyTelephony()
    vad = ProviderFactory.get_vad("energy")
    stt = ProviderFactory.get_stt("mock")
    llm = ProviderFactory.get_llm("mock")
    tts = ProviderFactory.get_tts("mock")

    session = CallSession(
        session_id="test-pipeline-run-1",
        telephony_adapter=telephony,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        db_session=async_db
    )

    await session.initialize()

    # Provide 1 second of dummy PCM audio
    dummy_audio = b"\x00" * 32000
    await session.run_cascaded_pipeline(dummy_audio)

    # Close session to flush queued writes
    await session.close()

    call = await CallRepository.get_call_details(async_db, session.db_call_id)
    assert call is not None
    assert call.status == "completed"
    # MockSTT produces 1 user transcript turn, MockLLM produces 1 assistant transcript turn
    assert len(call.transcripts) >= 2
    assert len(call.metrics) == 1
    assert call.metrics[0].turn_index == 1

