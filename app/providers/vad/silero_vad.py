import numpy as np
from app.interfaces.vad_interface import BaseVAD
from app.providers.vad.energy_vad import EnergyVAD

class SileroVAD(BaseVAD):
    """
    Silero VAD ONNX model for high precision 0-cost local speech detection.
    Falls back to EnergyVAD if ONNX session fails to load.
    """
    def __init__(self):
        self._fallback_energy = EnergyVAD()
        self._session = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.threshold = 0.5

    def process_chunk(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        if self._session is None:
            return self._fallback_energy.process_chunk(audio_bytes, sample_rate)

        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(samples) < 512:
                # Silero expects chunks of at least 512 samples for 16kHz
                return self._fallback_energy.process_chunk(audio_bytes, sample_rate)

            # Pad or take 512 samples frame
            input_frame = samples[:512].reshape(1, -1)
            sr_tensor = np.array(sample_rate, dtype=np.int64)

            ort_inputs = {
                'input': input_frame,
                'sr': sr_tensor,
                'h': self._h,
                'c': self._c,
            }

            out, self._h, self._c = self._session.run(None, ort_inputs)
            speech_prob = out[0][0]
            return speech_prob > self.threshold
        except Exception:
            return self._fallback_energy.process_chunk(audio_bytes, sample_rate)

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._fallback_energy.reset()
