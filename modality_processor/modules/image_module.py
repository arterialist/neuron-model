from typing import Generator, Any
import numpy as np
from PIL import Image

from modality_processor.core.base_module import BaseModule
from modality_processor.core.data_packet import DataPacket
from modality_processor.config import ImageModuleConfig
from modality_processor.core.policy import PolicyManager


class ImageModule(BaseModule):
    """
    A module for processing single image files.
    """

    def __init__(self, config: ImageModuleConfig, policy_manager: PolicyManager = None):  # type: ignore
        super().__init__(config, policy_manager)
        # Ensure config is of the correct type for static analysis
        self.config: ImageModuleConfig = config

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes image data to a float representation between 0 and 1.
        """
        if self.config.normalization:
            if data.dtype == np.uint8:
                return data.astype(np.float32) / 255.0
        return data

    def process_stream(self, source: str) -> Generator[DataPacket, None, None]:
        """
        Processes a single image file and yields one DataPacket.

        Args:
            source: The file path to the image.

        Yields:
            A single DataPacket containing the processed image data.
        """
        try:
            img = Image.open(source)

            # Convert to a consistent format (e.g., RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if specified in the config
            if self.config.output_resolution:
                img = img.resize(self.config.output_resolution)

            # Convert image to numpy array
            img_array = np.array(img)

            # Normalize the data
            normalized_data = self._normalize(img_array)

            # Reshape or process to fit output_tensor_size if needed
            if self.config.output_tensor_size:
                # Simple flattening, a real implementation might need more complex logic
                flat_data = normalized_data.flatten()
                if flat_data.size > self.config.output_tensor_size:
                    processed_data = flat_data[: self.config.output_tensor_size]
                else:
                    # Pad with zeros if the data is smaller than the tensor size
                    pad_width = self.config.output_tensor_size - flat_data.size
                    processed_data = np.pad(flat_data, (0, pad_width), "constant")
            else:
                processed_data = normalized_data

            # For a static image, timestamp can be 0
            timestamp = 0.0

            # A single image is treated as one "chunk"
            yield DataPacket(
                data=processed_data,
                modality=self.config.input_modality,
                timestamp=timestamp,
                metadata={"source": source, "shape": normalized_data.shape},
            )

        except FileNotFoundError:
            print(f"Error: Image file not found at {source}")
        except Exception as e:
            print(f"Error processing image {source}: {e}")
