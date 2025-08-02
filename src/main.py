from dotenv import load_dotenv
from api.api import create_app
from fastrtc import Stream
from fastapi import FastAPI
from app.flows.session_flow.stream import stream as session_stream

load_dotenv()

import click


@click.command()
@click.option(
    "--mode",
    default="ui",
    type=click.Choice(["server", "ui", "phone"], case_sensitive=False),
    help="Service to run: 'server' for FastAPI, 'ui' for Gradio UI, 'phone' for phone mode.",
)
def main(mode: str):
    stream: Stream | None = session_stream

    app: FastAPI = create_app(stream=stream)
    if mode == "ui":
        stream.ui.launch(
            server_port=7860,
            # server_name="127.0.0.1",
        )
    elif mode == "phone":
        stream.fastphone(host="0.0.0.0", port=7860)
    elif mode == "server":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)

    raise Exception(f"Not implemented mode '{mode}'")


if __name__ == "__main__":
    main()  # type: ignore
