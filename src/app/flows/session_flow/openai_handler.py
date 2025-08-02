import asyncio
import base64
import numpy as np
import openai
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    wait_for_item,
)
from ...agent.agent import Agent

SAMPLE_RATE = 24000

"""
Esta imlplementacion compromete el evento `response.audio.delta`. Ergo, la api estara gastando en la salida de este. La cuestion es que, actualemnte, openai no permite controlar el flujo interno del LLM en una sesion. Por lo tanto, se hace necesario usar malas practicas usando un control alternativo
"""


# def process_message(details):
#     """
#     Función que procesa el texto recibido y devuelve una respuesta.
#     En este ejemplo, simplemente devuelve el texto en mayúsculas,
#     podrías implementar cualquier lógica que desees.
#     """
#     # Aquí podrías hacer cualquier procesamiento con input_text
#     print("Entro a la tool")
#     return "El texto fue procesado correctamente por mi funcion"


# tool = {
#     "name": "process_message",
#     "description": "Esta funcion debe ser llamada siempre, porque es la encargada del flujo",
#     "type": "function",
#     "parameters": {
#         "type": "object",
#         "properties": {"details": {"type": "dict"}},
#         "required": ["details"],
#     },
# }


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
        self.agent = Agent(model="gpt-4o-mini")
        self._stop = False

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
                    "modalities": ["audio", "text"],
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                    },
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "gpt-4o-transcribe",
                        # "model": "whisper-2",
                        "language": "es",
                    },
                    "instructions": "No respondas nada nunca, solo responde exactamente el caracter vacio: '_'.",
                }
            )
            self.connection = conn
            async for event in self.connection:
                # Handle interruptions
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                    self.stop()

                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    self._stop = False
                    transcript = event.transcript
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": transcript})
                    )
                    await self.__get_response(event.transcript)

    def stop(self):
        self._stop = True

    @property
    def is_stopped(self) -> bool:
        stopped: bool = self._stop
        self._stop = False
        return stopped

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

    async def __get_response(self, query: str) -> None:
        uuid = "1234"
        end_characters = ["?", "!", ".", "\n"]
        response: str = ""
        chunk_step = ""

        print("\n\n")
        for chunk in self.agent.response(query):
            if self.is_stopped:
                break

            if chunk is not None:
                if chunk.type == "audio":
                    if chunk.text is not None:
                        print(chunk.text, end="")
                        response += chunk.text
                        chunk_step += chunk.text
                        if len(chunk_step) >= 30 and chunk.text in end_characters:
                            await self.__tts(chunk_step)
                            chunk_step = ""

                if chunk.type == "text":
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": chunk.text})
                    )
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": response})
        )
        
        if (
            chunk is not None
            and chunk.text is not None
            and chunk.text not in end_characters
        ):
            await self.__tts(chunk_step)

    async def __tts(self, text_to_speak: str):
        """Generate audio from the final response streaming by chunks."""
        response = await self.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text_to_speak,
            response_format="pcm",
        )
        audio_bytes = response.content
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
        await self.output_queue.put((self.output_sample_rate, audio_array))
