import inspect
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import hdf5_getters

# Collect all getter functions (except get_num_songs)
# all_getters = {
#     name: func
#     for name, func in inspect.getmembers(hdf5_getters, inspect.isfunction)
#     if name.startswith("get_") and name != "get_num_songs"
# }
useful_getters = [
    "get_artist_id",
    "get_artist_name",
    "get_song_id",
    "get_title",
    "get_duration",
    "get_year",
    "get_release",
    "get_artist_familiarity",
    "get_artist_hotttnesss",
    "get_song_hotttnesss",
    "get_artist_terms",
    "get_artist_terms_freq",
    "get_artist_terms_weight",
    "get_artist_location",
    "get_artist_latitude",
    "get_artist_longitude",
    "get_tempo",
    "get_similar_artists",
    "get_danceability",
    "get_energy",
    "get_loudness",
    "get_key",
    "get_mode",
]

all_getters = {
    name: func
    for name, func in inspect.getmembers(hdf5_getters, inspect.isfunction)
    if name in useful_getters
}


def extract_song_data(h5, songidx):
    """
    Extract all fields from the HDF5 file for one song.
    If a getter fails, we store None.
    If a getter returns an array, we convert it to a list.
    """
    row = {}

    for name, func in all_getters.items():
        try:
            value = func(h5, songidx)

            # Convert numpy arrays to Python lists
            if hasattr(value, "tolist"):
                value = value.tolist()

            row[name] = value

        except Exception as e:
            print(f"row[{name}] = None:", e)
            row[name] = None

    return row


rows = []

print("Scanning dataset...")

for root, _, files in os.walk("data/MillionSongSubset"):
    for f in files:
        if not f.endswith(".h5"):
            continue

        filepath = os.path.join(root, f)
        h5 = hdf5_getters.open_h5_file_read(filepath)

        num_songs = hdf5_getters.get_num_songs(h5)

        for songidx in range(num_songs):
            print(f"Processing file: {filepath}, song {songidx + 1}/{num_songs}")
            row = extract_song_data(h5, songidx)
            row["source_file"] = filepath  # useful for debugging
            rows.append(row)

        h5.close()


print("Building DataFrame...")
df = pd.DataFrame(rows)

print("Saving to Parquet...")
table = pa.Table.from_pandas(df)
pq.write_table(table, "data/MillionSongSubset.parquet")

print("Done!")
