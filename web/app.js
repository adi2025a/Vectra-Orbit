let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let audioQueue = [];
let isPlayingAudio = false;

const startCallBtn = document.getElementById("startCallBtn");
const stopCallBtn = document.getElementById("stopCallBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const waveformCanvas = document.getElementById("waveformCanvas");
const canvasCtx = waveformCanvas.getContext("2d");
const transcriptBox = document.getElementById("transcriptBox");
const transferAlert = document.getElementById("transferAlert");
const transferMsg = document.getElementById("transferMsg");
const callsTableBody = document.getElementById("callsTableBody");

let audioAnalyser = null;
let visualizerAnimId = null;

startCallBtn.addEventListener("click", startCall);
stopCallBtn.addEventListener("click", stopCall);

async function startCall() {
  try {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/call`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      updateStatus("Connected & Active", true);
      startCallBtn.disabled = true;
      stopCallBtn.disabled = false;
      transferAlert.style.display = "none";

      appendMessage("system", "AI Call Session connected. Start speaking into your microphone.");
      await setupAudioInput();
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        handleControlMessage(msg);
      } else if (event.data instanceof ArrayBuffer) {
        playAudioBuffer(event.data);
      }
    };

    ws.onclose = () => {
      stopCall();
    };

    ws.onerror = (err) => {
      console.error("WebSocket Error:", err);
      stopCall();
    };

  } catch (err) {
    alert("Could not access microphone: " + err.message);
  }
}

async function setupAudioInput() {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  
  const source = audioContext.createMediaStreamSource(mediaStream);
  audioAnalyser = audioContext.createAnalyser();
  audioAnalyser.fftSize = 64;
  source.connect(audioAnalyser);

  scriptProcessor = audioContext.createScriptProcessor(512, 1, 1);
  source.connect(scriptProcessor);
  scriptProcessor.connect(audioContext.destination);

  scriptProcessor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const floatSamples = e.inputBuffer.getChannelData(0);
    const pcm16 = floatTo16BitPCM(floatSamples);
    ws.send(pcm16.buffer);
  };

  drawWaveform();
}

function floatTo16BitPCM(float32Array) {
  const buffer = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return buffer;
}

function handleControlMessage(msg) {
  if (msg.event === "metrics") {
    updateMetrics(msg.data);
  } else if (msg.event === "transcript") {
    appendMessage(msg.role, msg.text);
  } else if (msg.event === "call_transferred") {
    transferAlert.style.display = "block";
    transferMsg.innerText = msg.message;
    appendMessage("system", `🔀 ${msg.message}`);
  } else if (msg.event === "call_ended") {
    stopCall();
  }
}

function updateMetrics(m) {
  document.getElementById("valVad").innerText = `${m.vad_latency_ms.toFixed(1)} ms`;
  document.getElementById("valStt").innerText = `${m.stt_latency_ms.toFixed(1)} ms`;
  document.getElementById("valLlm").innerText = `${m.llm_ttft_ms.toFixed(1)} ms`;
  document.getElementById("valTts").innerText = `${m.tts_first_chunk_ms.toFixed(1)} ms`;
  document.getElementById("valE2e").innerText = `${m.e2e_voice_latency_ms.toFixed(1)} ms`;
  document.getElementById("valBottleneck").innerText = m.max_latency_module || "None";

  document.getElementById("bottleneckText").innerText = 
    `Turn #${m.turn_id} Benchmark: E2E Latency ${m.e2e_voice_latency_ms.toFixed(1)}ms. Highest Latency Component: ${m.max_latency_module}`;
}

async function playAudioBuffer(arrayBuffer) {
  if (!audioContext) return;
  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    const source = audioContext.createBufferSource();
    source.buffer = decoded;
    source.connect(audioContext.destination);
    source.start(0);
  } catch (err) {
    // Binary raw chunk streaming fallback
  }
}

function stopCall() {
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }

  if (visualizerAnimId) {
    cancelAnimationFrame(visualizerAnimId);
  }

  startCallBtn.disabled = false;
  stopCallBtn.disabled = true;
  updateStatus("Disconnected", false);
  fetchCallHistory();
}

function updateStatus(text, active) {
  statusText.innerText = text;
  statusDot.className = `status-dot ${active ? "active" : ""}`;
}

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `transcript-message message-${role === "user" ? "user" : "ai"}`;
  
  const sender = document.createElement("div");
  sender.className = "message-sender";
  sender.innerText = role.toUpperCase();
  
  div.appendChild(sender);
  div.appendChild(document.createTextNode(text));
  
  transcriptBox.appendChild(div);
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function drawWaveform() {
  if (!audioAnalyser) return;
  const bufferLength = audioAnalyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    visualizerAnimId = requestAnimationFrame(render);
    audioAnalyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = "rgba(15, 23, 42, 0.4)";
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    const barWidth = (waveformCanvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * waveformCanvas.height;
      canvasCtx.fillStyle = `rgb(56, 189, 248)`;
      canvasCtx.fillRect(x, waveformCanvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 2;
    }
  }
  render();
}

async function fetchCallHistory() {
  try {
    const res = await fetch("/api/calls");
    if (!res.ok) return;
    const calls = await res.json();
    if (!calls || calls.length === 0) return;

    callsTableBody.innerHTML = "";
    calls.forEach(c => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${c.call_sid.substring(0, 8)}...</td>
        <td><span class="badge">${c.status}</span></td>
        <td>${c.duration_seconds ? c.duration_seconds.toFixed(1) + 's' : '--'}</td>
        <td>${c.avg_stt ? c.avg_stt.toFixed(1) + 'ms' : '--'}</td>
        <td>${c.avg_llm ? c.avg_llm.toFixed(1) + 'ms' : '--'}</td>
        <td>${c.avg_e2e ? c.avg_e2e.toFixed(1) + 'ms' : '--'}</td>
        <td>${c.was_transferred ? 'YES (' + c.transfer_target + ')' : 'No'}</td>
      `;
      callsTableBody.appendChild(tr);
    });
  } catch (err) {}
}

fetchCallHistory();
