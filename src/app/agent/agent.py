from openai import AsyncClient, Client
import os


class Agent:
    def __init__(self, model="gpt4o-mini"):
        self.client = Client(api_key=os.getenv("OPEN_API_KEY"))
        self.async_client = AsyncClient(api_key=os.getenv("OPEN_API_KEY"))
        self.model = model

    async def response(query: str):
        pass

    async def __flow(self):
        pass

    async def __task(self, query):
        system_message = {
            "role": "system",
            "content": "you are a good assistant that talk to users trough voice.",
        }
        user_message = {"role": "user", "content": query}
        messages = [system_message, user_message]
        stream = self.async_client.chat.completions.create(
            model="gpt4o-mini",
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            # Extract content from the chunk
            content = chunk.choices[0].delta.content
            yield content
