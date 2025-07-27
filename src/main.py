from dotenv import load_dotenv
from core.api import create_app
from fastrtc import Stream
from fastapi import FastAPI
from core.stream import create_stream
from core.session_flow.openai_handler import OpenAIHandler as OpenAIHandlerSession

load_dotenv()

import click


@click.command()
@click.option(
    "--flow",
    default="simple",
    type=click.Choice(["simple", "session"], case_sensitive=False),
    help="",
)
@click.option(
    "--mode",
    default="ui",
    type=click.Choice(["server", "ui", "phone"], case_sensitive=False),
    help="Service to run: 'server' for FastAPI, 'ui' for Gradio UI, 'phone' for phone mode.",
)
def main(flow: str, mode: str):
    stream: Stream | None = None
    if flow == "simple":
        stream = create_stream(OpenAIHandlerSession())
    if flow == "session":
        stream = create_stream(OpenAIHandlerSession())

    app: FastAPI = create_app(stream=stream)
    if mode == "ui":
        stream.ui.launch(server_port=7860)
    elif mode == "phone":
        stream.fastphone(host="0.0.0.0", port=7860)
    elif mode == "server":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)

    raise Exception("Not implemented mode")


if __name__ == "__main__":
    main()  # type: ignore
