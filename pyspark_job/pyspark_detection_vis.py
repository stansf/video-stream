import io
import os

import cv2
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from vs_detection import get_detector

from constants import KAFKA_SERVER, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC


@udf(BinaryType())
def f(im_bytes):
    image_file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()),
                                  dtype=np.uint8)
    img = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = get_detector()
    img_vis = detector.forward_vis(img)
    success, encoded_image = cv2.imencode('.jpeg', img_vis)
    result_bytes = encoded_image.tobytes()
    return result_bytes


def main():
    kafka_input_topic = os.getenv('KAFKA_INPUT_TOPIC', KAFKA_INPUT_TOPIC)
    kafka_output_topic = os.getenv('KAFKA_OUTPUT_TOPIC', KAFKA_OUTPUT_TOPIC)
    kafka_server = os.getenv('KAFKA_SERVER', KAFKA_SERVER)

    spark = SparkSession.builder.appName('Detection').\
        config('spark.driver.bindAddress', '0.0.0.0').\
        config('spark.ui.port', '4040').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    SparkContext.setLogLevel(spark.sparkContext, 'WARN')
    spark.conf.set('spark.sql.streaming.checkpointLocation',
                   './checkpoints')

    df = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', kafka_server) \
        .option('subscribe', kafka_input_topic).load()
    df2 = df.withColumn('vis', f(df.value))

    select_expr = 'vis as value'
    query = df2.selectExpr(select_expr).writeStream \
        .format('kafka') \
        .option('kafka.bootstrap.servers', kafka_server) \
        .option('topic', kafka_output_topic) \
        .start()

    query.awaitTermination()


if __name__ == '__main__':
    main()
