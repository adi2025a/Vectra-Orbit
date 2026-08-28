import json
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.repository import CallRepository
from app.db.models import CallAnalytics
from app.providers.factory import ProviderFactory

class CallAnalyticsEngine:
    """Async background worker that analyzes completed call transcripts."""

    @staticmethod
    async def analyze_call(db_session: AsyncSession, call_id: str):
        call = await CallRepository.get_call_details(db_session, call_id)
        if not call or not call.transcripts:
            return

        # Format complete dialogue transcript string
        transcript_text = "\n".join([
            f"{t.role.upper()}: {t.text}" for t in call.transcripts
        ])

        analysis_prompt = (
            "You are an expert AI Voice Call Analyst. Analyze the following phone call transcript and return a JSON object with:\n"
            "1. 'summary': A concise 2-sentence summary of the conversation.\n"
            "2. 'sentiment': One of ['positive', 'neutral', 'negative'].\n"
            "3. 'goal_achieved': boolean true/false indicating whether the call achieved its objective or successfully transferred.\n"
            "4. 'key_takeaways': list of 2-3 key points or follow-up action items.\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            "Respond ONLY with valid JSON."
        )

        llm = ProviderFactory.get_llm("groq")
        raw_response = ""
        async for chunk in llm.generate_response_stream(
            messages=[{"role": "user", "content": analysis_prompt}]
        ):
            if chunk.content:
                raw_response += chunk.content

        try:
            # Parse JSON analytics response
            data = json.loads(raw_response.strip())
            analytics = CallAnalytics(
                call_id=call.id,
                summary=data.get("summary", "Call completed."),
                sentiment=data.get("sentiment", "neutral"),
                goal_achieved=bool(data.get("goal_achieved", True)),
                key_takeaways=data.get("key_takeaways", [])
            )
            db_session.add(analytics)
            await db_session.commit()
            print(f"[Analytics] Successfully generated post-call analytics for Call ID {call_id}")
        except Exception as e:
            print(f"[Analytics Error] Failed to parse post-call analytics: {e}")
