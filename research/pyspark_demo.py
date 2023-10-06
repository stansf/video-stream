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


from constants import TOPIC, TOPIC2


@udf(BinaryType())
def img_echo(s):
    return s
    # print(str(type(s)))
    # return str(type(s))
    # print(s.value.decode('utf-8'))
    # return s.value.decode('utf-8').upper().encode('utf-8')

@udf
def f(im_bytes):
    # image_file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()),
    #                               dtype=np.uint8)
    # img = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
    # cv2.imwrite('img.png', img)
    n = 10
    coords = np.random.randint(100, size=(n, 4))
    classes = np.random.randint(10, size=n)
    results = dict(
        detections=[
            dict(label=lbl.tolist(), bbox=c.tolist())
            for lbl, c in zip(classes, coords)
        ]
    )
    return json.dumps(results)


def main():
    # conf = SparkConf()

    spark = SparkSession.builder.appName('Uppercase').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    SparkContext.setLogLevel(spark.sparkContext, 'WARN')
    spark.conf.set("spark.sql.streaming.checkpointLocation", './checkpoints')

    df = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', 'kafka:9092') \
        .option('subscribe', TOPIC).load()
    # query = df.rdd.map(f)

    # ff = udf(f, ByteType())

    # query = df.select('value').writeStream.format('console').foreach(lambda s: print(type(bytes(s.value)))).start()
    df2 = df.withColumn('value2', f(df.value))
    # query = df2.select('value2').writeStream.format('console').foreach(lambda s: print(len(s.value2))).start()
    # # query = df2.select('value2').writeStream.format('console').start()
    # # query = df2.select('value2').writeStream \


    select_expr = 'value2 as value'
    # select_expr = 'CAST(value2 as BINARY) as value'
    # query = df2.selectExpr('CAST(value2 as BYTES) as value').writeStream \
    query = df2.selectExpr(select_expr).writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("topic", TOPIC2) \
        .start()

    query.awaitTermination()


if __name__ == '__main__':
    main()
