from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, ByteType, BinaryType, StringType
# from pyspark.st
from pyspark.sql import Row
import numpy as np
import cv2
import io
import json

from detector import get_detector


from constants import TOPIC, TOPIC2


@udf
def f(im_bytes):
    image_file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()),
                                  dtype=np.uint8)
    img = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = get_detector()
    results = detector.forward(img)
    return json.dumps(results)


def main():
    spark = SparkSession.builder.appName('Uppercase').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    SparkContext.setLogLevel(spark.sparkContext, 'WARN')
    spark.conf.set("spark.sql.streaming.checkpointLocation", './checkpoints')

    df = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', 'kafka:9092') \
        .option('subscribe', TOPIC).load()
    df2 = df.withColumn('detections', f(df.value))

    select_expr = 'detections as value'
    query = df2.selectExpr(select_expr).writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("topic", TOPIC2) \
        .start()

    query.awaitTermination()


if __name__ == '__main__':
    main()
