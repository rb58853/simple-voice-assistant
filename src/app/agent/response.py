from pyparsing import Literal


class AgentResponse:
    def __init__(self, response_type: str, text: str):
        self.text = text
        self.type = response_type
