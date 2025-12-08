# Imports SparkSession - the main entry point to Spark
from pyspark.sql import SparkSession

# Imports helper functions: decode() for binary->string, col() to refer to columns
from pyspark.sql.functions import col, decode

# Creates (or retrieves) a SparkSession, which initializes Spark and lets us use the DataFrame API
spark = SparkSession.builder.appName("Show Titles").getOrCreate()

# Reads the Parquet file into a Spark DataFrame
df = spark.read.parquet("/data/MillionSongSubset.parquet")

# Replaces the existing column "get_title" with its decoded UTF-8 version
# here we overwrite the original column instead of creating a new one
df = df.withColumn("get_title", decode(col("get_title"), "UTF-8"))

# Selects the (already decoded) column and prints the first 20 rows without truncating long strings
df.select("get_title").show(20, truncate=False)
