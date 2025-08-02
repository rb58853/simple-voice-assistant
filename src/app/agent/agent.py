from openai import AsyncClient, Client
from typing import AsyncGenerator, Generator
import os
from .response import AgentResponse


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = Client(api_key=os.getenv("OPEN_API_KEY"))
        self.async_client = AsyncClient(api_key=os.getenv("OPEN_API_KEY"))
        self.model = model
        self.history: list[dict] = []

    def response(self, query: str) -> Generator[AgentResponse, None]:
        response: str = ""
        for step in self.__flow(query):
            if step.text is not None:
                if step.type == "audio":
                    response += step.text
                yield step
        self.history.append({"role": "assistant", "content": response})

    def __flow(self, query: str) -> Generator[AgentResponse, None]:
        yield AgentResponse("text", " • Procesando consulta")
        for step in self.__task(query):
            yield AgentResponse("audio", step)

    def __task(self, query) -> Generator[str, None]:
        system_message = {
            "role": "system",
            "content": "You are a friendly assistant.",
        }
        user_message = {"role": "user", "content": query}
        self.history.append(user_message)

        messages = [system_message] + self.history
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            # Extract content from the chunk
            content = chunk.choices[0].delta.content
            yield content
