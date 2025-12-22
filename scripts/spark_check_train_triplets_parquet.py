# Check the Taste Profile Subset Parquet file
# Print schema, row count, sample rows, basic stats, and null counts
#
# USAGE:
# docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/work-dir/scripts/spark_check_train_triplets_parquet.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum

spark = (
    SparkSession.builder
    .appName("Check Train Triplets Parquet")
    .getOrCreate()
)

PARQUET_PATH = "/opt/spark/work-dir/data/train_triplets.parquet"

df = spark.read.parquet(PARQUET_PATH)

print("\n=== SCHEMA ===")
df.printSchema()

print("\n=== ROW COUNT ===")
print(df.count())

print("\n=== SAMPLE ROWS ===")
df.show(10, truncate=False)

print("\n=== BASIC STATS ===")
df.select("play_count").summary().show()

print("\n=== TOP USERS ===")
df.groupBy("user_id") \
  .agg(_sum("play_count").alias("total_plays")) \
  .orderBy(col("total_plays").desc()) \
  .show(10, truncate=False)

print("\n=== TOP SONGS ===")
df.groupBy("song_id") \
  .agg(_sum("play_count").alias("total_plays")) \
  .orderBy(col("total_plays").desc()) \
  .show(10, truncate=False)

spark.stop()
