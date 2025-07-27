import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_twilio_turn_credentials,
)
from fastrtc.utils import audio_to_bytes
from gradio.utils import get_space
from pydantic import BaseModel
from app.speech.speech import SpeechClient
from openai import AsyncOpenAI
import asyncio

from ..agent.agent import Agent

agent: Agent = Agent()
speech_client = SpeechClient()
llm_client = AsyncOpenAI()
curr_dir = Path(__file__).parent


async def async_response(audio, chatbot=None):
    """Asynchronous response function with optimized streaming."""
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]

    # Process STT
    prompt = await speech_client.speech_to_text(
        ("audio-file.mp3", audio_to_bytes(audio))
    )
    chatbot.append({"role": "user", "content": prompt})
    yield AdditionalOutputs(chatbot)

    # Set up streaming response
    start = time.time()
    print("starting response pipeline", start)

    # Buffer for collecting the complete response
    complete_response = ""
    sentence_buffer = ""

    # Start LLM streaming
    stream = await agent.response(prompt)
    
    async for chunk in stream:
        # Extract content from the chunk
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        complete_response += content
        sentence_buffer += content

        # Check if we have a complete sentence or significant phrase
        if (
            "." in content or "!" in content or "?" in content or "\n" in content
        ) and len(sentence_buffer) > 15:
            # Process this sentence for TTS - use async for to iterate through yielded chunks
            async for audio_data in speech_client.text_to_speech_stream(
                sentence_buffer
            ):
                yield audio_data

            sentence_buffer = ""

    # Process any remaining text in the buffer
    if sentence_buffer:
        async for audio_data in speech_client.text_to_speech_stream(sentence_buffer):
            yield audio_data

    # Update chat history
    chatbot.append({"role": "assistant", "content": complete_response})
    yield AdditionalOutputs(chatbot)
    print("finished response pipeline", time.time() - start)

def sentence_buffer_to_audio(sentence_buffer:str):
    pass

def response(audio: tuple[int, np.ndarray], chatbot: list[dict] | None = None):
    """Synchronous wrapper for the asynchronous response generator."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agen = async_response(audio, chatbot)

        while True:
            try:
                # Get the next item from the async generator
                item = loop.run_until_complete(agen.__anext__())
                yield item
            except StopAsyncIteration:
                # Exit loop when the async generator is exhausted
                break
            except Exception as e:
                print(f"Error in response generator: {e}")
                # Continue with the next iteration rather than breaking completely
                continue
    finally:
        loop.close()


chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)