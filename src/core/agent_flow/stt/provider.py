import abc
import numpy as np
from typing import Optional


class ProviderSTT(abc.ABC):
    @abc.abstractmethod
    def initialize(self) -> None:
        """Prepare the provider (e.g., load model, set up API client)."""
        pass

    @abc.abstractmethod
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        **kwargs
    ) -> str:
        """Convert audio input to transcribed text."""
        pass

    def text_to_speech(
        self,
        text: str,
    ):
        pass
