import cv2
import numpy as np
from typing import Generator
from modality_processor.core.base_module import BaseModule
from modality_processor.core.data_packet import DataPacket
from modality_processor.config import VideoModuleConfig
from modality_processor.core.policy import PolicyManager

class VideoModule(BaseModule):
    """
    A module for processing video files, demonstrating streaming and multi-modality.
    It consults a PolicyManager to decide which modalities (video, audio) to process.
    """

    def __init__(self, config: VideoModuleConfig, policy_manager: PolicyManager):
        super().__init__(config, policy_manager)
        self.config: VideoModuleConfig = config

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalizes a video frame's pixel values to be between 0 and 1.
        """
        if self.config.normalization:
            if frame.dtype == np.uint8:
                return frame.astype(np.float32) / 255.0
        return frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single video frame (resize, normalize, etc.).
        """
        # Simplified resizing based on output_tensor_size
        if self.config.output_tensor_size:
            side_len = int(np.sqrt(self.config.output_tensor_size / 3))
            if side_len > 0:
                dim = (side_len, side_len)
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        normalized_frame = self._normalize_frame(frame)
        return normalized_frame.flatten()

    def _simulate_audio_stream(self, duration: float, fps: float) -> Generator[DataPacket, None, None]:
        """
        Generates a dummy audio stream synchronized with the video.
        This simulates extracting audio from the video file.
        """
        if not self.policy_manager.is_enabled("audio"):
            return

        # Simulate a 440 Hz sine wave for the audio
        sample_rate = 44100
        frequency = 440.0
        audio_chunk_size = 1024
        
        num_chunks = int(duration * sample_rate / audio_chunk_size)
        
        for i in range(num_chunks):
            timestamp = (i * audio_chunk_size) / sample_rate
            
            # Stop yielding audio if it gets ahead of the video timestamp
            if timestamp > duration:
                break

            # Generate a chunk of sine wave data
            t = np.linspace(timestamp, timestamp + audio_chunk_size / sample_rate, audio_chunk_size, endpoint=False)
            audio_data = np.sin(2. * np.pi * frequency * t)

            yield DataPacket(
                data=audio_data.astype(np.float32),
                modality="audio",
                timestamp=timestamp,
                metadata={'source': 'simulated_from_video', 'status': 'placeholder'}
            )

    def process_stream(self, source: str) -> Generator[DataPacket, None, None]:
        """
        Processes a video file, yielding video and/or audio packets based on policy.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # --- Audio Stream ---
        if self.config.extract_audio:
            yield from self._simulate_audio_stream(duration, fps)

        # --- Video Stream ---
        if self.policy_manager.is_enabled("video"):
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self._process_frame(frame_rgb)
                timestamp = frame_count / fps
                
                yield DataPacket(
                    data=processed_frame,
                    modality=self.config.input_modality,
                    timestamp=timestamp,
                    metadata={'source': source, 'frame_number': frame_count}
                )
                frame_count += 1

        cap.release()
