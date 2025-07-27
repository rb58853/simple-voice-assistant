import abc
import numpy as np
from typing import Generator, Tuple


class ProviderTTS(abc.ABC):
    """Base abstract class for TTS providers."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the TTS provider."""
        pass

    @abc.abstractmethod
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = None,
        language: str = None,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Convert text to speech."""
        pass
