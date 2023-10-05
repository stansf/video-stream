from time import sleep

from confluent_kafka import Producer

from constants import TOPIC


def main0():
    producer = Producer({'bootstrap.servers': 'localhost:9093'})
    for i in range(10):
        producer.produce(TOPIC, b'Hello')
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

    for i in range(10):
        data = f'Hello {i}'
        # Trigger any available delivery report callbacks from previous produce() calls
        p.poll(0.5)

        # Asynchronously produce a message. The delivery report callback will
        # be triggered from the call to poll() above, or flush() below, when the
        # message has been successfully delivered or failed permanently.
        p.produce(TOPIC, data.encode('utf-8'), callback=delivery_report)
        sleep(0.5)

    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    p.flush()


if __name__ == '__main__':
    main()
