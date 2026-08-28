# Vectra-Orbit 🎙️
### Enterprise Modular AI Voice Calling Platform (FastAPI + WebSockets + Cascaded Pipeline)

Vectra-Orbit is a real-time, cascaded AI voice calling platform built with **FastAPI**, **WebSockets**, and **SQLAlchemy (PostgreSQL / SQLite)**. It is designed to execute voice campaigns (marketing, surveys, customer support) with automatic call forwarding / transfer to real human customer care representatives.

---

## 🌟 Key Features

1. **Cascaded Decoupled Voice Pipeline**:
   - **VAD (Voice Activity Detection)**: `Silero VAD` (local ONNX, 0 cost) / `Energy VAD`
   - **STT (Speech-to-Text)**: `Groq Whisper` (Free cloud tier) / `faster-whisper` / `Mock`
   - **LLM (Dialogue Engine)**: `Groq Llama 3.3 70B` (Free cloud tier) / `Ollama` (Local) / `Mock`
   - **TTS (Text-to-Speech)**: `Microsoft Edge Neural TTS` (100% Free neural speech streaming)
   - **Telephony**: In-Browser Web Audio WebSocket Adapter + Twilio Media Streams & TwiML `<Dial>` Adapter.

2. **Granular Latency & Bottleneck Tracking**:
   - Measures precise millisecond latency for every turn:
     - `VAD Latency`
     - `STT Latency`
     - `LLM Time-To-First-Token (TTFT)`
     - `TTS First Audio Chunk Latency`
     - `E2E Voice Roundtrip Latency`
   - Automatically identifies and highlights the bottleneck component for every conversation turn.

3. **Barge-in (Interruption Handling)**:
   - Real-time VAD detects when human speech interrupts AI response playback, instantly flushing audio buffers and cancelling active LLM/TTS generation tasks.

4. **Human Customer Care Call Forwarding**:
   - When requested by the caller or required by campaign rules, the LLM emits a `transfer_call(target_phone_number)` tool call that signals the telephony layer to execute a SIP REFER / Twilio `<Dial>` handoff.

5. **PostgreSQL Conversation Database & Analytics**:
   - Stores full multi-turn transcript turns, structured `JSONB` metadata, and granular metric records.
   - Includes async post-call worker for sentiment analysis, objective evaluation, and transcript summaries.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your free Groq API key to `.env` for sub-200ms cloud STT & LLM, or set `STT_PROVIDER=mock` and `LLM_PROVIDER=mock` for 100% offline testing).*

### 3. Launch Server & Web Dashboard
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to test live full-duplex calls using your microphone and speaker!

---

## 🛠️ Testing

Run the automated unit & integration test suite:
```bash
PYTHONPATH=. pytest tests/
```
