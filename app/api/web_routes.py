import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.session import get_db
from app.db.repository import CallRepository

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(BASE_DIR, "web")

@router.get("/", response_class=FileResponse)
async def serve_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Index HTML not found")

@router.get("/api/calls")
async def get_recent_calls(db: AsyncSession = Depends(get_db)):
    """Fetch recent call logs and aggregated latency stats for UI table."""
    calls = await CallRepository.list_recent_calls(db, limit=20)
    result = []
    for c in calls:
        metrics = c.metrics or []
        total_turns = len(metrics)
        avg_stt = sum(m.stt_latency_ms for m in metrics) / total_turns if total_turns > 0 else 0.0
        avg_llm = sum(m.llm_ttft_ms for m in metrics) / total_turns if total_turns > 0 else 0.0
        avg_e2e = sum(m.e2e_voice_latency_ms for m in metrics) / total_turns if total_turns > 0 else 0.0
        
        result.append({
            "id": c.id,
            "call_sid": c.call_sid,
            "status": c.status,
            "duration_seconds": c.duration_seconds,
            "was_transferred": c.was_transferred,
            "transfer_target": c.transfer_target,
            "start_time": c.start_time.isoformat() if c.start_time else None,
            "avg_stt": avg_stt,
            "avg_llm": avg_llm,
            "avg_e2e": avg_e2e,
            "turns_count": total_turns
        })
    return result
