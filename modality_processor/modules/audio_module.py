import numpy as np
import soundfile as sf
from typing import Generator

from modality_processor.core.base_module import BaseModule
from modality_processor.core.data_packet import DataPacket
from modality_processor.config import AudioModuleConfig
from modality_processor.core.policy import PolicyManager


class AudioModule(BaseModule):
    """
    A module for processing audio files.
    """

    def __init__(self, config: AudioModuleConfig, policy_manager: PolicyManager = None):  # type: ignore
        super().__init__(config, policy_manager)
        self.config: AudioModuleConfig = config

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes audio data to be between -1 and 1.
        Assumes data is floating point.
        """
        if self.config.normalization:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                return data / max_val
        return data

    def process_stream(self, source: str) -> Generator[DataPacket, None, None]:
        """
        Processes an audio file, chunking it and yielding DataPackets.

        Args:
            source: The file path to the audio file.

        Yields:
            A stream of DataPackets, each containing a chunk of audio data.
        """
        try:
            with sf.SoundFile(source, "r") as audio_file:
                samplerate = audio_file.samplerate

                # --- Placeholder for resampling if needed ---
                # if self.config.sample_rate != samplerate:
                #   # Resampling logic would go here. Requires a library like `librosa` or `scipy.signal`.
                #   # For now, we just print a warning.
                #   print(f"Warning: Audio file sample rate ({samplerate} Hz) differs from config ({self.config.sample_rate} Hz). No resampling is performed.")
                # --- End Placeholder ---

                chunk_size_frames = self.config.chunk_size
                timestamp_counter = 0

                for block in audio_file.blocks(blocksize=chunk_size_frames):
                    # Normalize the audio chunk
                    processed_data = self._normalize(block)

                    # If the audio is stereo, we might want to convert it to mono
                    if len(processed_data.shape) > 1 and processed_data.shape[1] > 1:
                        processed_data = np.mean(processed_data, axis=1)

                    # Calculate timestamp in seconds
                    timestamp = (timestamp_counter * chunk_size_frames) / samplerate

                    yield DataPacket(
                        data=processed_data,
                        modality=self.config.input_modality,
                        timestamp=timestamp,
                        metadata={"source": source, "samplerate": samplerate},
                    )
                    timestamp_counter += 1

        except FileNotFoundError:
            print(f"Error: Audio file not found at {source}")
        except Exception as e:
            print(f"Error processing audio file {source}: {e}")
