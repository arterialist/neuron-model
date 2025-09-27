from abc import ABC, abstractmethod
from typing import Generator, Any
from modality_processor.core.data_packet import DataPacket
from modality_processor.config import BaseModuleConfig
from modality_processor.core.policy import PolicyManager


class BaseModule(ABC):
    """
    Abstract base class for a modality processing module.

    Each module is responsible for handling a specific data modality (e.g., video, text).
    It must implement a stream processing method that takes raw data and yields
    standardized DataPacket objects.
    """

    def __init__(self, config: BaseModuleConfig, policy_manager: PolicyManager = None):  # type: ignore
        self.config = config
        self.policy_manager = policy_manager or PolicyManager()

    @abstractmethod
    def process_stream(self, source: Any) -> Generator[DataPacket, None, None]:
        """
        Processes a data source as a stream and yields DataPackets.

        This method should be implemented as a generator.

        Args:
            source: The data source to process. This could be a file path,
                    a network stream, or any other data handle.

        Yields:
            DataPacket: A standardized packet of processed data.
        """
        pass

    def _normalize(self, data: Any) -> Any:
        """
        Performs normalization on the data if enabled in the config.
        This is a placeholder and should be overridden by subclasses with
        modality-specific normalization logic.
        """
        if self.config.normalization:
            # Basic normalization, subclasses should implement more specific logic
            # For example, for images, this would be scaling pixel values.
            # For audio, it might be amplitude scaling.
            pass
        return data
