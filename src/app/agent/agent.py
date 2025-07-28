from openai import AsyncClient, Client
from typing import AsyncGenerator, Generator
import os


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = Client(api_key=os.getenv("OPEN_API_KEY"))
        self.async_client = AsyncClient(api_key=os.getenv("OPEN_API_KEY"))
        self.model = model
        self.history: list[dict] = []

    async def response(self, query: str) -> AsyncGenerator[str, None]:
        response: str = ""
        async for step in self.__flow(query):
            if step is not None:
                response += step
                yield step
        self.history.append({"role": "assistant", "content": response})

    async def __flow(self, query: str) -> AsyncGenerator[str, None]:
        async for step in self.__task(query):
            yield step

    async def __task(self, query) -> AsyncGenerator[str, None]:
        system_message = {
            "role": "system",
            "content": "you are a good assistant that talk to users trough voice.",
        }
        user_message = {"role": "user", "content": query}
        self.history.append(user_message)

        messages = [system_message] + self.history
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            # Extract content from the chunk
            content = chunk.choices[0].delta.content
            yield content
