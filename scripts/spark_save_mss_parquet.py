# Convert Million Song Dataset HDF5 files to Parquet format
#
# USAGE:
# docker exec -it spark-master /opt/spark/bin/spark-submit --py-files /opt/spark/work-dir/scripts/hdf5_getters.py /opt/spark/work-dir/scripts/spark_save_mss_parquet.py
#
# ========================================================
# IMPORTANT:
# Before running this script, make sure the host folder ./data
# (mounted as /opt/spark/work-dir/data inside the container)
# has write permissions for all users. You can do this on the host with:
# chmod 777 ./data
# This ensures Spark inside the container can create the Parquet file.
# ========================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)
import os
import numpy as np

# --------------------------------------------------------
# Spark session
# --------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Save MSS Metadata to Parquet")
    .getOrCreate()
)

# --------------------------------------------------------
# Add the hdf5_getters.py file to all workers
# --------------------------------------------------------
sc = spark.sparkContext
sc.addPyFile("/opt/spark/work-dir/scripts/hdf5_getters.py")  # Путь к hdf5_getters.py

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
MSS_ROOT = "/opt/spark/work-dir/data/MillionSongSubset"
OUTPUT_PATH = "/opt/spark/work-dir/data/MillionSongSubset.parquet"

# --------------------------------------------------------
# Explicit schema (required for Spark)
# --------------------------------------------------------
schema = StructType([
    StructField("song_id", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("artist_id", StringType(), True),
    StructField("artist_name", StringType(), True),
    StructField("title", StringType(), True),
    StructField("release", StringType(), True),
    StructField("artist_familiarity", DoubleType(), True),
    StructField("artist_hotttnesss", DoubleType(), True),
    StructField("artist_location", StringType(), True),
    StructField("song_hotttnesss", DoubleType(), True),
    StructField("year", IntegerType(), True),
    StructField("duration", DoubleType(), True),
    StructField("tempo", DoubleType(), True),
    StructField("loudness", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("danceability", DoubleType(), True),
    StructField("key", IntegerType(), True),
    StructField("mode", IntegerType(), True),
    StructField("time_signature", IntegerType(), True),
])

# --------------------------------------------------------
# Step 1: Find all .h5 files recursively (distributed)
# --------------------------------------------------------
# Use binaryFile with proper path handling to avoid URI scheme issues
h5_files_rdd = (
    spark.read
    .format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(MSS_ROOT)
    .select("path")
    .rdd
    .map(lambda r: r.path)
    .filter(lambda p: p.endswith(".h5"))
    # Fix path by removing file:// prefix if present
    .map(lambda p: p[7:] if p.startswith("file://") else (p[5:] if p.startswith("file:") else p))
)

print("Number of .h5 files found:", h5_files_rdd.count())

# --------------------------------------------------------
# Step 2: Parse HDF5 file (function to be used in workers)
# --------------------------------------------------------
def parse_h5_file(path: str):
    import hdf5_getters as h5g
    
    # Ensure path is clean (remove file:// prefix if present)
    if isinstance(path, str) and path.startswith("file:"):
        if path.startswith("file://"):
            path = path[7:]  # Remove "file://"
        else:
            path = path[5:]  # Remove "file:"

    try:
        h5 = h5g.open_h5_file_read(path)

        # Helper function to convert numpy types to Python native types
        def convert_value(value):
            if hasattr(value, 'item'):  # numpy scalar
                return value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                return value.tolist()
            else:
                return value

        row = (
            convert_value(h5g.get_song_id(h5)),
            convert_value(h5g.get_track_id(h5)),
            convert_value(h5g.get_artist_id(h5)),
            convert_value(h5g.get_artist_name(h5)),
            convert_value(h5g.get_title(h5)),
            convert_value(h5g.get_release(h5)),
            float(convert_value(h5g.get_artist_familiarity(h5))),
            float(convert_value(h5g.get_artist_hotttnesss(h5))),
            convert_value(h5g.get_artist_location(h5)),
            float(convert_value(h5g.get_song_hotttnesss(h5))),
            int(convert_value(h5g.get_year(h5))),
            float(convert_value(h5g.get_duration(h5))),
            float(convert_value(h5g.get_tempo(h5))),
            float(convert_value(h5g.get_loudness(h5))),
            float(convert_value(h5g.get_energy(h5))),
            float(convert_value(h5g.get_danceability(h5))),
            int(convert_value(h5g.get_key(h5))),
            int(convert_value(h5g.get_mode(h5))),
            int(convert_value(h5g.get_time_signature(h5))),
        )

        h5.close()
        return row

    except Exception as e:
        # Skip corrupted or problematic files
        print(f"Failed to parse {path}: {e}")
        return None

# --------------------------------------------------------
# Step 3: Parse all files in parallel
# --------------------------------------------------------
songs_rdd = (
    h5_files_rdd
    .map(parse_h5_file)
    .filter(lambda x: x is not None)
)

print("Number of extracted records:", songs_rdd.count())

# --------------------------------------------------------
# Step 4: Create DataFrame with explicit schema
# --------------------------------------------------------
df = spark.createDataFrame(songs_rdd, schema=schema)

# --------------------------------------------------------
# Step 5: Write to Parquet
# --------------------------------------------------------
df.write.mode("overwrite").parquet(OUTPUT_PATH)

print("MSS metadata Parquet file saved!")

# Stop Spark session
spark.stop()
