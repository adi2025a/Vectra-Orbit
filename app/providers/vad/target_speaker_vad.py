import numpy as np
from typing import Optional
from app.interfaces.vad_interface import BaseVAD
from app.providers.vad.silero_vad import SileroVAD
from app.config import settings


class TargetSpeakerVAD(BaseVAD):
    """
    Target Speaker Voice Activity Detector (TS-VAD).
    
    Acts as a modular 2-Stage Filter:
    - Stage 1: Standard VAD (Silero or Energy) gates speech vs silence (~20ms latency).
    - Stage 2: Dynamic Speaker Embedding Matching. Automatically locks onto the primary
      caller's acoustic embedding signature during the initial speech segment (~1.0s)
      and ignores background TV speech, radio, or non-target voices in subsequent frames.
    """

    def __init__(
        self,
        base_vad: Optional[BaseVAD] = None,
        threshold: float = settings.TARGET_SPEAKER_THRESHOLD,
        enroll_seconds: float = settings.TARGET_SPEAKER_ENROLL_SECONDS,
    ):
        self.base_vad = base_vad if base_vad is not None else SileroVAD()
        self.threshold = threshold
        self.enroll_seconds = enroll_seconds

        self.target_embedding: Optional[np.ndarray] = None
        self.is_enrolled: bool = False
        self.enrollment_buffer = bytearray()
        self.window_buffer = bytearray()

    def _extract_embedding(self, pcm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a normalized d-vector speaker embedding from PCM 16-bit mono audio.
        Uses Mel-scale log-filterbank spectral energy distribution & dynamic-range
        centered acoustic profiling to form a speaker fingerprint.
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) < 256:
            return np.zeros(64, dtype=np.float32)

        # Compute STFT / Spectrogram representation (Frame length 512, Hop length 256)
        n_fft = 512
        hop_length = 256
        n_frames = 1 + (len(samples) - n_fft) // hop_length
        if n_frames < 1:
            samples = np.pad(samples, (0, max(0, n_fft - len(samples))))
            n_frames = 1

        window = np.hanning(n_fft)
        frames = np.lib.stride_tricks.sliding_window_view(samples[: (n_frames - 1) * hop_length + n_fft], n_fft)[::hop_length]
        
        # Spectrogram magnitude
        spectrogram = np.abs(np.fft.rfft(frames * window, n=n_fft))
        
        # 64 Mel-scale filterbanks
        n_mels = 64
        mel_basis = self._create_mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
        mel_spectrogram = np.dot(spectrogram, mel_basis.T)

        # Dynamic range floor & relative log energy scaling to avoid zero-floor bias
        peak_energy = np.max(mel_spectrogram)
        if peak_energy > 1e-6:
            log_mel = np.log10(np.maximum(mel_spectrogram, peak_energy * 1e-3))
            log_mel = log_mel - np.mean(log_mel, axis=-1, keepdims=True)
        else:
            log_mel = np.zeros_like(mel_spectrogram)

        # Average across time frames to get speaker d-vector
        embedding = np.mean(log_mel, axis=0)

        # L2 Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            embedding = np.zeros(n_mels, dtype=np.float32)

        return embedding

    def _create_mel_filterbank(self, n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Helper to construct triangular Mel filterbank matrix."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        low_freq_mel = hz_to_mel(0)
        high_freq_mel = hz_to_mel(sample_rate / 2.0)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

        for i in range(1, n_mels + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            if center > left:
                filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        return filterbank

    def process_chunk(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        """
        Process a chunk of audio (PCM 16-bit).
        1. Run base VAD gate. If no speech detected by base VAD, return False.
        2. If base VAD detects speech:
           - If not enrolled, accumulate audio until target speaker profile is locked.
           - If enrolled, calculate cosine similarity against target embedding.
        """
        # Stage 1: Fast Base VAD Gate
        is_speech = self.base_vad.process_chunk(audio_bytes, sample_rate)
        if not is_speech:
            return False

        # Stage 2: Speaker Embedding Verification
        bytes_per_sec = sample_rate * 2  # 16-bit = 2 bytes per sample
        required_enroll_bytes = int(self.enroll_seconds * bytes_per_sec)

        if not self.is_enrolled:
            self.enrollment_buffer.extend(audio_bytes)
            if len(self.enrollment_buffer) >= required_enroll_bytes:
                # Lock Target Speaker Profile from initial speech segment
                self.target_embedding = self._extract_embedding(bytes(self.enrollment_buffer), sample_rate)
                self.is_enrolled = True
            # Allow speech during initial enrollment turn
            return True

        # Enrolled State: Verify chunk against locked Target Speaker Embedding
        self.window_buffer.extend(audio_bytes)
        eval_window_bytes = int(0.25 * bytes_per_sec)  # 250ms window

        if len(self.window_buffer) >= eval_window_bytes:
            eval_bytes = bytes(self.window_buffer)
            # Keep trailing 100ms overlap for sliding window continuity
            overlap_bytes = int(0.10 * bytes_per_sec)
            self.window_buffer = self.window_buffer[-overlap_bytes:]

            chunk_embedding = self._extract_embedding(eval_bytes, sample_rate)
            if self.target_embedding is not None:
                # Cosine Similarity: dot product of normalized vectors
                similarity = float(np.dot(chunk_embedding, self.target_embedding))
                return similarity >= self.threshold

        return True

    def set_target_embedding(self, embedding: np.ndarray) -> None:
        """Manually pre-set or override the target speaker embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            self.target_embedding = embedding / norm
        else:
            self.target_embedding = embedding
        self.is_enrolled = True

    def reset(self) -> None:
        """Reset internal buffers and VAD states (keeps target_embedding intact unless full reset)."""
        self.window_buffer.clear()
        self.base_vad.reset()

    def full_reset(self) -> None:
        """Full reset between different call sessions (clears locked target speaker profile)."""
        self.target_embedding = None
        self.is_enrolled = False
        self.enrollment_buffer.clear()
        self.window_buffer.clear()
        self.base_vad.reset()
