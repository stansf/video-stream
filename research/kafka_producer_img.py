from time import sleep

import cv2
from confluent_kafka import Producer

from constants import TOPIC_IMAGES


def main0():
    producer = Producer({'bootstrap.servers': 'localhost:9093'})
    for i in range(10):
        producer.produce(TOPIC_IMAGES, b'Hello')
        print('Send', i+1)
        sleep(0.8)


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

    i = 0
    while True:
        img = cv2.imread('image.png')
        img_enc = cv2.imencode('.png', img)
        # print(img_enc[1].shape)
        # print(img_enc[1].tobytes())
        img_b = img_enc[1].tobytes()
        print(type(img_b), len(img_b))
        # data = f'Hello {i}'
        # Trigger any available delivery report callbacks from previous produce() calls
        p.poll(0.5)

        # Asynchronously produce a message. The delivery report callback will
        # be triggered from the call to poll() above, or flush() below, when the
        # message has been successfully delivered or failed permanently.
        p.produce(TOPIC_IMAGES, img_b, callback=delivery_report)
        sleep(5)
        i += 1

    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    p.flush()


if __name__ == '__main__':
    main()
