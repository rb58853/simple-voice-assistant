import asyncio
import base64
import numpy as np
import openai
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    wait_for_item,
)

SAMPLE_RATE = 24000

"""
Esta imlplementacion compromete el evento `response.audio.delta`. Ergo, la api estara gastando en la salida de este. La cuestion es que, actualemnte, openai no permite controlar el flujo interno del LLM en una sesion. Por lo tanto, se hace necesario usar malas practicas usando un control alternativo
"""


class OpenAIHandler(AsyncStreamHandler):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()

    def copy(self):
        return OpenAIHandler()

    async def start_up(
        self,
    ):
        """Connect to realtime API. Run forever in separate thread to keep connection open."""
        self.client = openai.AsyncOpenAI()
        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview-2024-12-17"
        ) as conn:
            await conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "es",
                    },
                    
                }
            )
            self.connection = conn
            async for event in self.connection:
                # Handle interruptions
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                if event.type == "response.audio_transcript.done":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )
                if event.type == "response.audio.delta":
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(
                                base64.b64decode(event.delta), dtype=np.int16
                            ).reshape(1, -1),
                        ),
                    )

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_message)  # type: ignore

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None
