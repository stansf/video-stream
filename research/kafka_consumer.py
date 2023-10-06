from confluent_kafka import Consumer

from constants import TOPIC, TOPIC2


def main():
    consumer = Consumer({'bootstrap.servers': 'localhost:9093',
                         'group.id': 'mygroup'})
    consumer.subscribe([TOPIC2])
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue

        print('Received message: {}'.format(msg.value().decode('utf-8')))


if __name__ == '__main__':
    main()
