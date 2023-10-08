import os
import time
from time import sleep

import cv2
from confluent_kafka import Producer

from .capture import RTSPVideoWriterObject
from .constants import KAFKA_TOPIC, STREAM_URL2, KAFKA_SERVER


def delivery_report(err, msg) -> None:
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')


def run_stream() -> None:
    kafka_server = os.getenv('KAFKA_SERVER', KAFKA_SERVER)
    print('Use kafka server:', kafka_server)

    stream_url = os.getenv('STREAM_URL', STREAM_URL2)
    print('Stream source:', stream_url)

    p = Producer({'bootstrap.servers': kafka_server})
    streamer = RTSPVideoWriterObject(stream_url)

    kafka_topic = os.getenv('KAFKA_TOPIC', KAFKA_TOPIC)
    print('Topic:', kafka_topic)

    timeout = os.getenv('TIMEOUT', None)

    while True:
        img = streamer.get_frame()
        if img is None:
            print('wait...')
            sleep(1)
            continue
        img_enc = cv2.imencode('.jpeg', img)
        img_b = img_enc[1].tobytes()
        # Trigger any available delivery report callbacks from previous
        # produce() calls
        p.poll(0.5)

        # Asynchronously produce a message. The delivery report callback will
        # be triggered from the call to poll() above, or flush() below, when
        # the message has been successfully delivered or failed permanently.
        p.produce(kafka_topic, img_b, callback=delivery_report)
        if timeout:
            time.sleep(float(timeout))


    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    p.flush()


if __name__ == '__main__':
    run_stream()
