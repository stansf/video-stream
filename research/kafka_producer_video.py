from time import sleep

import cv2
from confluent_kafka import Producer

from constants import TOPIC_IMAGES, stream_url
from capture import RTSPVideoWriterObject


def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(),
                                                    msg.partition()))


def main():
    p = Producer({'bootstrap.servers': 'localhost:9093'})
    streamer = RTSPVideoWriterObject(stream_url)

    i = 0
    while True:
        img = streamer.get_frame()
        if img is None:
            print('wait...')
            sleep(1)
            continue
        img_enc = cv2.imencode('.jpeg', img)
        img_b = img_enc[1].tobytes()
        # Trigger any available delivery report callbacks from previous produce() calls
        p.poll(0.5)

        # Asynchronously produce a message. The delivery report callback will
        # be triggered from the call to poll() above, or flush() below, when the
        # message has been successfully delivered or failed permanently.
        p.produce(TOPIC_IMAGES, img_b, callback=delivery_report)
        # sleep(5)
        i += 1

    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    p.flush()


if __name__ == '__main__':
    main()
