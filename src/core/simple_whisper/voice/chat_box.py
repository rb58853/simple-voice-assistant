import gradio as gr
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from gradio.utils import get_space

def update_chatbot(chatbot: list[dict], response: dict):
        chatbot.append(response)
        return chatbot
    
class ChatBox:
    def __init__(self):
        self.chatbot = gr.Chatbot(type="messages")
        self.additional_inputs = [self.chatbot]
        self.additional_outputs = [self.chatbot]
        self.additional_outputs_handler = update_chatbot

    async def event_functions(self, event, handler: AsyncStreamHandler):
        # Handle interruptions
        if event.type == "input_audio_buffer.speech_started":
            handler.clear_queue()
        if event.type == "conversation.item.input_audio_transcription.completed":
            await handler.output_queue.put(
                AdditionalOutputs({"role": "user", "content": event.transcript})
            )
        if event.type == "response.audio_transcript.done":
            await handler.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": event.transcript})
            )


class NoneChatBox(ChatBox):
    def __init__(self):
        self.chatbot = None
        self.additional_inputs = None
        self.additional_outputs = None
        self.additional_outputs_handler = None

    def update_chatbot(self, chatbot: list[dict], response: dict):
        pass

    async def event_functions(self, event, handler: AsyncStreamHandler):
        pass
