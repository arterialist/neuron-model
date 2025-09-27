from typing import Dict, Type, List, Any
from modality_processor.core.base_module import BaseModule, BaseModuleConfig
from modality_processor.core.data_packet import DataPacket
from modality_processor.core.policy import PolicyManager
import os

class ModalityProcessor:
    """
    The central orchestrator for processing various data modalities.

    This class manages a collection of modality-specific modules. It inspects
    incoming data, selects the appropriate module(s), and delegates the
    processing task to them.
    """

    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._modality_map: Dict[str, str] = {} # Maps file extension to modality
        self.policy_manager = PolicyManager()

    def set_policy(self, policy: Dict[str, Any]):
        """
        Sets the processing policy for the PolicyManager.

        Args:
            policy: A dictionary defining the policy (e.g., {'enabled_modalities': ['video', 'audio']}).
        """
        self.policy_manager.set_policy(policy)
        print(f"Policy updated. Enabled modalities: {self.policy_manager._enabled_modalities}")

    def register_module(self, module_class: Type[BaseModule], config: BaseModuleConfig):
        """
        Registers a modality processing module with the processor.

        Args:
            module_class: The class of the module to register (e.g., ImageModule).
            config: The configuration object for the module.
        """
        module_instance = module_class(config, self.policy_manager)
        modality = config.input_modality
        if modality in self._modules:
            print(f"Warning: Module for modality '{modality}' is being replaced.")
        
        self._modules[modality] = module_instance
        
        if hasattr(config, 'supported_containers'):
            for ext in config.supported_containers: # type: ignore
                self._modality_map[ext] = modality
        print(f"Registered module for modality: '{modality}'")

    def process(self, source: str):
        """
        Processes a data source (e.g., a file path).

        It determines the data modality and uses the corresponding module
        to process the data, yielding a stream of DataPackets.

        Args:
            source: The data source, typically a file path.

        Yields:
            DataPacket: A stream of processed data packets from the module.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")

        _, ext = os.path.splitext(source)
        modality = self._modality_map.get(ext.lower())

        if not modality:
            raise ValueError(f"No module registered for file type '{ext}'")

        module = self._modules.get(modality)
        if not module:
            # This case should ideally not be reached if _modality_map is built correctly
            raise ValueError(f"Module for modality '{modality}' not found, though a mapping exists.")

        print(f"Processing '{source}' with '{module.__class__.__name__}'...")
        yield from module.process_stream(source)

