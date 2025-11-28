import os
from scripts import hdf5_getters, item_based_cf

# Path to the Million Song Subset HDF5 files
MSS_PATH = "data/MillionSongSubset"


def find_h5_file_for_song(song_id):
    """
    Search through the MillionSongSubset directory for a .h5 file containing the given song_id.

    Args:
        song_id (str): The ID of the song to search for.

    Returns:
        str or None: Path to the HDF5 file if found, otherwise None.
    """
    # Walk through all directories and files in MSS_PATH
    for root, _, files in os.walk(MSS_PATH):
        for file in files:
            if file.endswith(".h5"):
                path = os.path.join(root, file)
                try:
                    # Open HDF5 file in read mode
                    with hdf5_getters.open_h5_file_read(path) as h:
                        # Get number of songs in this file
                        num_songs = hdf5_getters.get_num_songs(h)
                        # Check each song in the file
                        for idx in range(num_songs):
                            if song_id == hdf5_getters.get_song_id(h, idx).decode(
                                "utf-8"
                            ):
                                return path, idx
                except Exception as e:
                    print(e)
                    # Skip files that cannot be opened or read
                    continue
    return None


def get_song_info(song_id):
    """
    Retrieve the artist name and song title for a given song_id.

    Args:
        song_id (str): The ID of the song.

    Returns:
        tuple: (artist_name, song_title). If not found, returns (song_id, song_id).
    """
    result = find_h5_file_for_song(song_id)
    if result is None:
        return song_id, song_id  # Return IDs if song not found

    h5_file, song_idx = result

    # Open the file and extract artist and title
    with hdf5_getters.open_h5_file_read(h5_file) as h:
        artist = hdf5_getters.get_artist_name(h, song_idx).decode("utf-8")
        title = hdf5_getters.get_title(h, song_idx).decode("utf-8")
        return artist, title


def main():
    """
    Main function to get song recommendations for a specific user.
    Fetches recommended song IDs using item-based collaborative filtering,
    then prints artist and title for each recommendation.
    """
    # Example user ID for whom to generate recommendations
    user_id = "b80344d063b5ccb3212f76538f3d9e43d87dca9e"

    # Get top N recommended song IDs
    recommended_ids = item_based_cf.recommend_songs(user_id, n_recommendations=3)

    # Retrieve artist and title for each recommended song
    recommendations = [get_song_info(sid) for sid in recommended_ids]

    # Print the recommendations
    print("Recommendations for user:", user_id)
    for i, (artist, title) in enumerate(recommendations, start=1):  # pyright: ignore
        if artist == title:
            # No info found - print song_id only once
            print(f"{i}. [song_id] {artist}")
        else:
            # Normal case - artist and title found
            print(f"{i}. {artist} - {title}")


if __name__ == "__main__":
    main()
