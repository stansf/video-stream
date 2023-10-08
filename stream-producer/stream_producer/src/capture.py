from threading import Thread
from typing import Optional

import cv2

from .constants import STREAM_URL2


class RTSPVideoWriterObject(object):
    def __init__(
            self: 'RTSPVideoWriterObject',
            src: str | int = 0,
            out_video_file: Optional[str] = None):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.output_video = cv2.VideoWriter(
            out_video_file, self.codec, 30,
            (self.frame_width, self.frame_height)
        ) if out_video_file else None
        self.frame = None

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self: 'RTSPVideoWriterObject'):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self: 'RTSPVideoWriterObject'):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            if self.output_video:
                self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self: 'RTSPVideoWriterObject'):
        assert self.output_video is not None, 'Out video is not set.'
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

    def get_frame(self: 'RTSPVideoWriterObject'):
        return self.frame


if __name__ == '__main__':
    rtsp_stream_link = STREAM_URL2
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)
    while True:
        try:
            video_stream_widget.show_frame()
            #video_stream_widget.save_frame()
        except AttributeError:
            pass
