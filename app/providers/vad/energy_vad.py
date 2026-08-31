import math
import numpy as np
from app.interfaces.vad_interface import BaseVAD


class EnergyVAD(BaseVAD):
    """
    Lightweight Energy/RMS-based Voice Activity Detector (0 external dependencies).
    Analyzes PCM 16-bit audio signal energy against dynamic threshold.
    """
    def __init__(self, threshold_db: float = -35.0):
        self.threshold_db = threshold_db

    def process_chunk(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        if not audio_bytes:
            return False
        # Convert PCM 16-bit signed to numpy array
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) == 0:
            return False
        rms = np.sqrt(np.mean(samples**2) + 1e-12)
        db = 20 * math.log10(rms + 1e-12)
        return db > self.threshold_db

    def reset(self) -> None:
        pass
