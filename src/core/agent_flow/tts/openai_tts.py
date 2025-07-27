import os
import numpy as np
from openai import OpenAI
from fastrtc import audio_to_bytes
import io
from .provider import ProviderTTS


class OpenAITTS(ProviderTTS):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model_id = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
        self.default_language = os.getenv("OPENAI_STT_LANGUAGE", "es")
        self.client = OpenAI(api_key=self.api_key)
        self.initialized = True

    def initialize(self) -> None:
        pass  # already initialized

    def text_to_speech(self, text, **kwargs):
        pass
