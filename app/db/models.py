import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import List, Optional

class Base(DeclarativeBase):
    pass

class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    objective_type: Mapped[str] = mapped_column(String(50), default="marketing")  # marketing, survey, support
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    transfer_phone_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    voice_name: Mapped[str] = mapped_column(String(100), default="en-US-AvaNeural")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    calls: Mapped[List["Call"]] = relationship("Call", back_populates="campaign")

class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("campaigns.id"), nullable=True)
    call_sid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    telephony_provider: Mapped[str] = mapped_column(String(50), default="web")
    customer_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="initiated")  # initiated, active, transferred, completed, failed
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    was_transferred: Mapped[bool] = mapped_column(Boolean, default=False)
    transfer_target: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    transfer_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    campaign: Mapped[Optional[Campaign]] = relationship("Campaign", back_populates="calls")
    transcripts: Mapped[List["CallTranscript"]] = relationship("CallTranscript", back_populates="call", cascade="all, delete-orphan")
    metrics: Mapped[List["CallMetric"]] = relationship("CallMetric", back_populates="call", cascade="all, delete-orphan")
    analytics: Mapped[Optional["CallAnalytics"]] = relationship("CallAnalytics", back_populates="call", uselist=False, cascade="all, delete-orphan")

class CallTranscript(Base):
    __tablename__ = "call_transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    call_id: Mapped[str] = mapped_column(String(36), ForeignKey("calls.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" or "assistant" or "system"
    text: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    call: Mapped["Call"] = relationship("Call", back_populates="transcripts")

class CallMetric(Base):
    __tablename__ = "call_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    call_id: Mapped[str] = mapped_column(String(36), ForeignKey("calls.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vad_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    stt_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_ttft_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tts_first_chunk_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tts_total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    e2e_voice_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    max_latency_module: Mapped[str] = mapped_column(String(100), default="N/A")

    call: Mapped["Call"] = relationship("Call", back_populates="metrics")

class CallAnalytics(Base):
    __tablename__ = "call_analytics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    call_id: Mapped[str] = mapped_column(String(36), ForeignKey("calls.id"), nullable=False, unique=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # positive, neutral, negative
    goal_achieved: Mapped[bool] = mapped_column(Boolean, default=False)
    key_takeaways: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    call: Mapped["Call"] = relationship("Call", back_populates="analytics")
