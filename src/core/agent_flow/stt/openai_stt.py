import os
import numpy as np
from openai import OpenAI
from fastrtc import audio_to_bytes
import io
from .provider import ProviderSTT


class OpenAISTT(ProviderSTT):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model_id = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
        self.default_language = os.getenv("OPENAI_STT_LANGUAGE", "es")
        self.client = OpenAI(api_key=self.api_key)
        self.initialized = True

    def initialize(self) -> None:
        pass  # already initialized

    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        prompt: str = None,
        temperature: float = 0.0,
        response_format: str = "json",
        **kwargs
    ) -> str:
        audio_bytes = audio_to_bytes(audio)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"  # required by OpenAI

        transcript = self.client.audio.transcriptions.create(
            model=self.default_model_id,
            file=audio_file,
            language=self.default_language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )

        return transcript.text

    def text_to_speech(
        self,
        text: str,
    ):
        pass
