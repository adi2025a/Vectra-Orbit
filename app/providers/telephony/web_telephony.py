import json
from typing import Dict, Any
from fastapi import WebSocket
from app.interfaces.telephony_interface import BaseTelephonyAdapter

class WebTelephonyAdapter(BaseTelephonyAdapter):
    """
    Browser Web Audio WebSocket Adapter.
    Enables live end-to-end full-duplex calls directly in the browser via microphone and speaker.
    """
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def parse_incoming_event(self, raw_payload: Any) -> Dict[str, Any]:
        """
        Accepts binary audio PCM data or JSON control events.
        """
        if isinstance(raw_payload, bytes):
            return {
                "event": "media",
                "media": {
                    "payload": raw_payload
                }
            }
        elif isinstance(raw_payload, str):
            try:
                data = json.loads(raw_payload)
                return data
            except Exception:
                return {"event": "unknown", "raw": raw_payload}
        return {"event": "unknown"}

    async def send_audio_chunk(self, session_id: str, audio_bytes: bytes) -> None:
        """
        Sends raw binary audio frames or base64 frames to the Web browser client.
        """
        if audio_bytes and self.websocket:
            await self.websocket.send_bytes(audio_bytes)

    async def transfer_call(self, session_id: str, target_phone_number: str) -> bool:
        """
        Notify the browser UI that call forwarding / transfer to human support was triggered.
        """
        if self.websocket:
            event = {
                "event": "call_transferred",
                "target_phone_number": target_phone_number,
                "message": f"Forwarding call to human customer support representative at {target_phone_number}..."
            }
            await self.websocket.send_text(json.dumps(event))
            return True
        return False

    async def end_call(self, session_id: str, reason: str = "completed") -> None:
        if self.websocket:
            event = {
                "event": "call_ended",
                "reason": reason
            }
            try:
                await self.websocket.send_text(json.dumps(event))
            except Exception:
                pass


class TwilioTelephonyAdapter(BaseTelephonyAdapter):
    """
    Twilio Media Streams & SIP Transfer Telephony Adapter.
    Parses mulaw 8kHz audio streams and executes TwiML `<Dial>` call transfers.
    """
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def parse_incoming_event(self, raw_payload: Any) -> Dict[str, Any]:
        if isinstance(raw_payload, str):
            data = json.loads(raw_payload)
            event_type = data.get("event")
            if event_type == "media":
                import base64
                payload_b64 = data["media"]["payload"]
                raw_mulaw = base64.b64decode(payload_b64)
                return {
                    "event": "media",
                    "media": {"payload": raw_mulaw},
                    "stream_sid": data.get("streamSid")
                }
            return data
        return {"event": "unknown"}

    async def send_audio_chunk(self, session_id: str, audio_bytes: bytes) -> None:
        if self.websocket and audio_bytes:
            import base64
            payload_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            msg = {
                "event": "media",
                "streamSid": session_id,
                "media": {"payload": payload_b64}
            }
            await self.websocket.send_text(json.dumps(msg))

    async def transfer_call(self, session_id: str, target_phone_number: str) -> bool:
        """
        Sends Twilio TwiML `<Dial>` command payload to forward the active phone call.
        """
        if self.websocket:
            msg = {
                "event": "transfer",
                "streamSid": session_id,
                "twiml": f"<Response><Dial>{target_phone_number}</Dial></Response>"
            }
            await self.websocket.send_text(json.dumps(msg))
            return True
        return False

    async def end_call(self, session_id: str, reason: str = "completed") -> None:
        if self.websocket:
            msg = {"event": "stop", "streamSid": session_id}
            try:
                await self.websocket.send_text(json.dumps(msg))
            except Exception:
                pass
