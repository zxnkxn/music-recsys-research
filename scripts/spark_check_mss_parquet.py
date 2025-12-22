# Check the Million Song Dataset Parquet file
# Print schema, row count, sample rows, basic stats, and null counts
#
# USAGE:
# docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/work-dir/scripts/spark_check_mss_parquet.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("Check MSS Parquet")
    .getOrCreate()
)

PARQUET_PATH = "/opt/spark/work-dir/data/MillionSongSubset.parquet"

df = spark.read.parquet(PARQUET_PATH)

print("\n=== SCHEMA ===")
df.printSchema()

print("\n=== ROW COUNT ===")
print(df.count())

print("\n=== SAMPLE ROWS ===")
df.show(10, truncate=False)

print("\n=== BASIC STATS ===")
df.select(
    "duration",
    "tempo",
    "loudness",
    "year"
).summary().show()

print("\n=== NULL COUNTS ===")
df.select([
    col(c).isNull().cast("int").alias(c)
    for c in df.columns
]).groupBy().sum().show(truncate=False)

spark.stop()
