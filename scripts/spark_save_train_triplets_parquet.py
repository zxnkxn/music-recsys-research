# Convert Taste Profile Subset TSV file to Parquet format
#
# USAGE:
# docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/work-dir/scripts/spark_save_train_triplets_parquet.py
#
# ========================================================
# IMPORTANT:
# Before running this script, make sure the host folder ./data
# (mounted as /opt/spark/work-dir/data inside the container)
# has write permissions for all users. You can do this on the host with:
# chmod 777 ./data
# This ensures Spark inside the container can create the Parquet file.
# ========================================================

# Import SparkSession to create a Spark context
from pyspark.sql import SparkSession

# Import col for column operations in DataFrame
from pyspark.sql.functions import col

# Import IntegerType to cast columns to integer
from pyspark.sql.types import IntegerType

# Create (or reuse) a SparkSession
# This is the entry point to work with Spark SQL and DataFrames
spark = SparkSession.builder.appName("Save Taste Profile to Parquet").getOrCreate()

# Read the Taste Profile Subset (TSV file) into a Spark DataFrame
# Spark will read the file in parallel across worker nodes
# Rename the columns to "user_id", "song_id", "play_count"
# Convert "play_count" column from string to integer
df = (
    spark.read.option("sep", "\t")
    .csv("/opt/spark/work-dir/data/train_triplets.txt")
    .toDF("user_id", "song_id", "play_count")
    .withColumn("play_count", col("play_count").cast(IntegerType()))
)

# Write the DataFrame to Parquet format
# mode("overwrite") ensures that if the file already exists, it will be replaced
# Parquet is a columnar format that is faster and more efficient for large datasets
df.write.mode("overwrite").parquet(
    "/opt/spark/work-dir/data/train_triplets.parquet"
)

# Print confirmation message
print("Parquet file saved!")