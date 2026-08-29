from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTelephonyAdapter(ABC):
    """Abstract Interface for Telephony Integration (WebSockets / Twilio / Plivo / SIP)."""

    @abstractmethod
    async def parse_incoming_event(self, raw_payload: Any) -> Dict[str, Any]:
        """
        Parse raw incoming telephony protocol payloads into a normalized CallEvent.
        """
        pass

    @abstractmethod
    async def send_audio_chunk(self, session_id: str, audio_bytes: bytes) -> None:
        """
        Send audio frame back to the telephony carrier or browser.
        """
        pass

    @abstractmethod
    async def transfer_call(self, session_id: str, target_phone_number: str) -> bool:
        """
        Execute call forwarding / transfer to a human customer care representative or SIP address.
        """
        pass

    @abstractmethod
    async def end_call(self, session_id: str, reason: str = "completed") -> None:
        """
        Hang up / terminate call session.
        """
        pass

    async def send_control_event(self, event_dict: Dict[str, Any]) -> None:
        """
        Send JSON control / message event to telephony client or browser.
        """
        pass
