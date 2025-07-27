from collections.abc import Generator
from typing import Tuple
from fastrtc import AlgoOptions, ReplyOnPause, Stream
import numpy as np

from .stt.openai_stt import OpenAISTT, ProviderSTT
from .agent import Agent

agent: Agent = Agent()
speech_service: ProviderSTT = OpenAISTT()


class AgentStream:
    def __init__(
        self, agent: Agent = Agent(), speech_service: ProviderSTT = OpenAISTT()
    ):
        self.agent: Agent = agent
        self.speech_service: ProviderSTT = speech_service

    @property
    def stream(self) -> Stream:
        return Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                self.__response,
                algo_options=AlgoOptions(speech_threshold=0.2),
            ),
        )

    def __response(
        self,
        audio: tuple[int, np.ndarray],
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        # STT: Audio -> Text
        transcript = self.speech_service.speech_to_text(audio)

        # Agent: Text -> Text
        agent_response = self.agent.invoke(transcript)
        response_text = agent_response

        # TTS: Text -> Audio
        yield from self.speech_service.text_to_speech(
            response_text,
        )

