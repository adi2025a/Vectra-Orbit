import uuid
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.factory import ProviderFactory
from app.core.session import CallSession
from app.core.analytics import CallAnalyticsEngine

router = APIRouter()

@router.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())

    # Initialize configured modular components
    vad = ProviderFactory.get_vad()
    stt = ProviderFactory.get_stt()
    llm = ProviderFactory.get_llm()
    tts = ProviderFactory.get_tts()
    telephony = ProviderFactory.get_telephony(websocket)

    session = CallSession(
        session_id=session_id,
        telephony_adapter=telephony,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts
    )
    await session.initialize()

    print(f"[WebSocket] Connected call session {session_id}")

    try:
        while True:
            # Receive binary audio chunks or text events from client
            raw_message = await websocket.receive()
            
            if "bytes" in raw_message and raw_message["bytes"]:
                pcm_chunk = raw_message["bytes"]
                await session.process_incoming_audio(pcm_chunk)
                
            elif "text" in raw_message and raw_message["text"]:
                event = await telephony.parse_incoming_event(raw_message["text"])
                if event.get("event") == "media":
                    payload = event["media"]["payload"]
                    await session.process_incoming_audio(payload)
                elif event.get("event") == "stop":
                    break

    except WebSocketDisconnect:
        print(f"[WebSocket] Disconnected call session {session_id}")
    except Exception as e:
        print(f"[WebSocket Error] Exception in session {session_id}: {e}")
    finally:
        await session.close()
        # Trigger background post-call analytics worker
        if session.db_call_id:
            asyncio.create_task(CallAnalyticsEngine.analyze_call(session.db_call_id))
