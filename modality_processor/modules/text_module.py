import numpy as np
from typing import Generator, List
from modality_processor.core.base_module import BaseModule
from modality_processor.core.data_packet import DataPacket
from modality_processor.config import TextModuleConfig
from modality_processor.core.policy import PolicyManager

class TextModule(BaseModule):
    """
    A module for processing text files.
    """

    def __init__(self, config: TextModuleConfig, policy_manager: PolicyManager = None):
        super().__init__(config, policy_manager)
        self.config: TextModuleConfig = config
        # Simple vocabulary for demonstration purposes
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()

    def _build_vocab(self):
        # In a real scenario, this would be loaded from a file or pre-trained tokenizer
        # For this example, we'll create a simple character-level vocabulary
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,\n"
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for i, char in enumerate(chars)}

    def _normalize(self, text: str) -> str:
        """
        Normalizes text data (e.g., converts to lowercase).
        """
        if self.config.normalization:
            return text.lower()
        return text

    def _text_to_int(self, text: str) -> List[int]:
        """
        Converts a string of text to a list of integers based on the vocabulary.
        """
        return [self.vocab.get(char, 0) for char in text] # Default to 0 for unknown chars

    def process_stream(self, source: str) -> Generator[DataPacket, None, None]:
        """
        Processes a text file, chunking it and converting to numerical format.

        Args:
            source: The file path to the text file.

        Yields:
            A stream of DataPackets, each containing a chunk of processed text data.
        """
        try:
            with open(source, 'r', encoding=self.config.encoding) as f:
                content = f.read()

            normalized_text = self._normalize(content)
            
            # This is a placeholder for tokenization. A real implementation
            # would use a proper tokenizer (e.g., from Hugging Face, NLTK).
            if self.config.chunk_by_words:
                tokens = normalized_text.split()
            else: # Chunk by characters
                tokens = list(normalized_text)

            timestamp_counter = 0
            for i in range(0, len(tokens), self.config.chunk_size):
                chunk = tokens[i:i + self.config.chunk_size]
                
                # Convert chunk to numerical representation
                # For this example, we'll join words back and convert char by char
                chunk_text = " ".join(chunk) if self.config.chunk_by_words else "".join(chunk)
                numerical_data = self._text_to_int(chunk_text)
                
                # Pad or truncate to fit output_tensor_size if specified
                if self.config.output_tensor_size:
                    if len(numerical_data) > self.config.output_tensor_size:
                        processed_data = numerical_data[:self.config.output_tensor_size]
                    else:
                        pad_width = self.config.output_tensor_size - len(numerical_data)
                        processed_data = numerical_data + [0] * pad_width # Pad with 0
                else:
                    processed_data = numerical_data

                yield DataPacket(
                    data=np.array(processed_data, dtype=np.int32),
                    modality=self.config.input_modality,
                    timestamp=float(timestamp_counter),
                    metadata={'source': source, 'chunk_index': timestamp_counter}
                )
                timestamp_counter += 1

        except FileNotFoundError:
            print(f"Error: Text file not found at {source}")
        except Exception as e:
            print(f"Error processing text file {source}: {e}")
