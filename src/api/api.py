import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import get_twilio_turn_credentials
from gradio.utils import get_space
from pathlib import Path
from fastrtc import Stream

cur_dir = Path(__file__).parent


def create_app(stream: Stream) -> FastAPI:
    app: FastAPI = FastAPI()
    stream.mount(app)

    @app.get("/")
    async def _():
        rtc_config = get_twilio_turn_credentials() if get_space() else None
        html_content = (cur_dir / "index.html").read_text()
        html_content = html_content.replace(
            "__RTC_CONFIGURATION__", json.dumps(rtc_config)
        )
        return HTMLResponse(content=html_content)

    @app.get("/outputs")
    def _(webrtc_id: str):
        async def output_stream():
            import json

            async for output in stream.output_stream(webrtc_id):
                s = json.dumps(output.args[0])
                yield f"event: output\ndata: {s}\n\n"

        return StreamingResponse(output_stream(), media_type="text/event-stream")

    return app
