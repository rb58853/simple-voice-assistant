from typing import Literal
from openai import AsyncOpenAI
from pydantic import SecretStr
import numpy as np


class SpeechClient:
    def __init__(
        self,
        stt_model: str = "whisper-1",
        tts_model: str = "kokoro",
        voice: str = "coral",
        tts_backend: str = "kokoro",
        tts_audio_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "pcm",
        stt_response_format: Literal["json", "text", "str"] = "json",
        language: str = "en",
    ):
        self.__stt_model = stt_model
        self.__stt_response_format = stt_response_format
        self.__language = language
        self.__tts_model = tts_model
        self.__voice = voice
        self.__tts_backend = tts_backend
        self.__tts_audio_format = tts_audio_format

        # Initialize async clients
        self.__tts_client = AsyncOpenAI()
        self.__stt_client = AsyncOpenAI()

    async def speech_to_text(self, audio_file: tuple) -> str:
        """Asynchronous version of speech-to-text conversion"""
        response = await self.__stt_client.audio.transcriptions.create(
            model=self.__stt_model,
            file=audio_file,
            language=self.__language,
            response_format=self.__stt_response_format,
        )
        return response.text

    async def text_to_speech_stream(self, text: str):
        """Process a text chunk and yield audio data sequentially.

        This function creates a streaming TTS response and yields each audio chunk
        as it becomes available, allowing for real-time audio playback.
        """
        try:
            async with self.__tts_client.audio.speech.with_streaming_response.create(
                model=self.__tts_model,
                voice=self.__voice,
                input=text,
                response_format=self.__tts_audio_format,
                extra_body={"backend": self.__tts_backend, "language": self.__language},
            ) as stream_audio:
                # Iterate through all audio chunks in the stream
                print("\nProcessing audio chunk...")
                async for audio_chunk in stream_audio.iter_bytes(chunk_size=1024):
                    print(".", end="")
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(
                        1, -1
                    )
                    yield (24000, audio_array)
                    # audio_array = np.frombuffer(audio_chunk, dtype=np.uint8).reshape(1, -1)
                    # yield (48000, audio_array)
            print()
        except Exception as e:
            print(f"Error in TTS processing: {e}")
