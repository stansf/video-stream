import json
from pprint import pprint

from confluent_kafka import Consumer

from constants import TOPIC_DETECTIONS


def main():
    consumer = Consumer({'bootstrap.servers': 'localhost:9093',
                         'group.id': 'mygroup'})
    consumer.subscribe([TOPIC_DETECTIONS])
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue

        text = msg.value().decode('utf-8')
        print(type(text), text)
        try:
            results = json.loads(text)
            print('Received detections:')
            pprint(results)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
