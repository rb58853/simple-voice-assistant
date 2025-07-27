import gradio as gr
from fastrtc import Stream, get_twilio_turn_credentials
from gradio.utils import get_space
from fastrtc.tracks import HandlerType
from .openai_handler import OpenAIHandler

SAMPLE_RATE = 24000


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)


def create_stream(handler: HandlerType) -> Stream:
    stream = Stream(
        handler,
        mode="send-receive",
        modality="audio",
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
        additional_outputs_handler=update_chatbot,
        rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
        concurrency_limit=5 if get_space() else None,
        time_limit=90 if get_space() else None,
    )
    return stream


stream: Stream = create_stream(OpenAIHandler())
