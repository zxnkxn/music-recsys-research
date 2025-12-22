# Entry point for working with Spark SQL and DataFrame API
from pyspark.sql import SparkSession

# Provides column expressions for DataFrame operations
from pyspark.sql.functions import col

# Defines integer data type for schema casting
from pyspark.sql.types import IntegerType

# Create (or reuse) a SparkSession.
# This initializes the connection to the Spark cluster.
spark = SparkSession.builder.appName("Load Taste Profile").getOrCreate()

# Read the Taste Profile Subset (TSV file) into a Spark DataFrame.
# Spark reads the file in parallel across worker nodes.
df = (
    spark.read.option("sep", "\t")
    .csv("/opt/spark/work-dir/data/train_triplets.txt")
    .toDF("user_id", "song_id", "play_count")
)

# Convert play_count from string to integer
df = df.withColumn("play_count", col("play_count").cast(IntegerType()))

# Print the schema to inspect column names and data types
df.printSchema()

# Trigger execution and display a small sample of the data
df.show(5)

# Count the total number of user–song interactions in the dataset
print("Total interactions:", df.count())

# Count the number of unique users
print("Unique users:", df.select("user_id").distinct().count())

# Count the number of unique songs
print("Unique songs:", df.select("song_id").distinct().count())