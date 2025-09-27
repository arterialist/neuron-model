from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class BaseModuleConfig:
    """
    Base configuration for all modality modules.
    """
    input_modality: str
    normalization: bool = True
    chunk_size: int = 1024
    output_tensor_size: Optional[int] = None

@dataclass
class ImageModuleConfig(BaseModuleConfig):
    """
    Configuration specific to the ImageModule.
    """
    input_modality: str = "image"
    supported_containers: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    output_resolution: Optional[Tuple[int, int]] = None # (width, height)

@dataclass
class VideoModuleConfig(BaseModuleConfig):
    """
    Configuration specific to the VideoModule.
    """
    input_modality: str = "video"
    supported_containers: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov"])
    extract_audio: bool = True
    frame_rate: Optional[int] = None # Target frame rate for processing

@dataclass
class AudioModuleConfig(BaseModuleConfig):
    """
    Configuration specific to the AudioModule.
    """
    input_modality: str = "audio"
    supported_containers: List[str] = field(default_factory=lambda: [".wav", ".mp3"])
    sample_rate: int = 44100

@dataclass
class TextModuleConfig(BaseModuleConfig):
    """
    Configuration specific to the TextModule.
    """
    input_modality: str = "text"
    supported_containers: List[str] = field(default_factory=lambda: [".txt"])
    encoding: str = "utf-8"
    chunk_by_words: bool = True # If False, chunks by characters
