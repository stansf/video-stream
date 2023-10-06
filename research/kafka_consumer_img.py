from confluent_kafka import Consumer
import cv2
import numpy as np
import io
from constants import TOPIC, TOPIC2
import sys


def main():
    consumer = Consumer({'bootstrap.servers': 'localhost:9093',
                         'group.id': 'mygroup',
                         'auto.offset.reset': 'latest',
                         })
    consumer.subscribe([TOPIC2])
    i = 0
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue

        print('Received message.')
        im_bytes = msg.value()
        print(sys.getsizeof(im_bytes))
        print(im_bytes.decode('utf-8'))
        image_file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()),
                                      dtype=np.uint8)
        img = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(f'image_get_{i}.png', img)
        i += 1


if __name__ == '__main__':
    main()
