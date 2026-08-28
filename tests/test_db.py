import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.db.models import Base
from app.db.repository import CallRepository
from app.core.metrics import TurnMetric

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

@pytest.mark.asyncio
async def test_create_and_update_call(async_session):
    call = await CallRepository.create_call(
        session=async_session,
        call_sid="test-sid-123",
        telephony_provider="web"
    )
    assert call.id is not None
    assert call.status == "active"

    # Add transcript turn
    t = await CallRepository.add_transcript_turn(
        session=async_session,
        call_id=call.id,
        turn_index=1,
        role="user",
        text="Can you transfer me to support?"
    )
    assert t.id is not None

    # Record turn metric
    metric = TurnMetric(
        turn_id=1,
        vad_latency_ms=2.0,
        stt_latency_ms=120.0,
        llm_ttft_ms=50.0,
        tts_first_chunk_ms=30.0,
        e2e_voice_latency_ms=200.0,
        max_latency_module="STT (120.0ms)"
    )
    m = await CallRepository.record_turn_metric(async_session, call.id, metric)
    assert m.id is not None
    assert m.stt_latency_ms == 120.0

    # Mark transferred
    await CallRepository.mark_call_completed(
        async_session,
        call.id,
        status="transferred",
        was_transferred=True,
        transfer_target="+18005550199",
        transfer_reason="Customer request"
    )

    updated_call = await CallRepository.get_call_details(async_session, call.id)
    assert updated_call.status == "transferred"
    assert updated_call.was_transferred is True
    assert len(updated_call.transcripts) == 1
    assert len(updated_call.metrics) == 1
