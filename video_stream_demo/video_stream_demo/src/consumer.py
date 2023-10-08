from confluent_kafka import Consumer


def stream_frames(kafka_server, kafka_topic):
    consumer = Consumer({'bootstrap.servers': kafka_server,
                         'group.id': 'detections'})
    consumer.subscribe([kafka_topic])
    i = 0
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        im_bytes = msg.value()
        print(f'msg recieved: {i}')
        i += 1
        # time.sleep(1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + im_bytes + b'\r\n')
