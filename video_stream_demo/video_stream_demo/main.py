import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.requests import Request
import threading
from functools import partial
from fastapi.templating import Jinja2Templates

from .src import stream_frames, KAFKA_TOPIC, KAFKA_SERVER


HTTP_PORT = 6064
lock = threading.Lock()
app = FastAPI()

manager = None
count_keep_alive = 0

width = 1280
height = 720
templates = Jinja2Templates(directory=(Path(__file__).parent / 'templates'))

streamer = partial(stream_frames,
                   kafka_server=os.getenv('KAFKA_SERVER', KAFKA_SERVER),
                   kafka_topic=os.getenv('KAFKA_TOPIC', KAFKA_TOPIC))


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(
        "images_stream.html", context={"request": request}
    )


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(
        streamer(), media_type='multipart/x-mixed-replace; boundary=frame'
    )
