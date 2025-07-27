from pyparsing import Literal


class AgentResponse:
    def __init__(self, response_type: Literal["text", "audio"], text: str):
        self.text = text
        self.type = response_type
