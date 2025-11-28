import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. Load data
# ============================================================
print("Loading train_triplets.txt...")
taste_df = pd.read_csv(
    "data/train_triplets.txt",
    sep="\t",
    names=["user", "song", "play_count"],
)

# ============================================================
# 2. Encode user_id and song_id to indices
# ============================================================
print("Encoding user and song IDs...")

user_ids = taste_df["user"].unique()
song_ids = taste_df["song"].unique()

user_to_index = {u: i for i, u in enumerate(user_ids)}
song_to_index = {s: i for i, s in enumerate(song_ids)}

# reverse mapping to get song_id back
index_to_song = np.array(song_ids)

# ============================================================
# 3. Build COO → CSR sparse matrix
# ============================================================
print("Creating sparse matrix...")

row = taste_df["user"].map(user_to_index).values
col = taste_df["song"].map(song_to_index).values
data = taste_df["play_count"].values

matrix_coo = coo_matrix((data, (row, col)), shape=(len(user_ids), len(song_ids)))

# Convert to CSR format (efficient for row slicing)
user_song_matrix = matrix_coo.tocsr()

print("Matrix created.")
print("Shape:", user_song_matrix.shape)
print(
    "Density:",
    user_song_matrix.nnz / (user_song_matrix.shape[0] * user_song_matrix.shape[1]),
)

# ============================================================
# 4. Train KNN model for item-based CF
# ============================================================
print("Training NearestNeighbors...")

model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
# Fit on the transposed matrix so that rows = songs
model_knn.fit(user_song_matrix.T)

print("Model trained.")


# ============================================================
# 5. Recommendation function
# ============================================================
def recommend_songs(user_id, n_recommendations=5, k_neighbors=10):
    """
    Item-based CF: recommend songs similar to what the user has listened to.
    """
    if user_id not in user_to_index:
        raise ValueError("User ID not found in the dataset.")

    user_idx = user_to_index[user_id]

    # Get all songs the user has listened to
    user_vector = user_song_matrix[user_idx]  # sparse row
    listened_song_indices = user_vector.indices

    scores = {}

    # Iterate over all songs the user has listened to
    for song_idx in listened_song_indices:
        song_vector = user_song_matrix[:, song_idx].T  # column vector (song)

        distances, indices = model_knn.kneighbors(
            song_vector,
            n_neighbors=k_neighbors + 1,
        )

        for dist, neigh_idx in zip(distances[0], indices[0]):
            if neigh_idx == song_idx:
                continue

            song_id = index_to_song[neigh_idx]
            similarity = 1 - dist

            scores[song_id] = scores.get(song_id, 0) + similarity

    # Sort recommendations by score descending
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [song for song, score in recommended[:n_recommendations]]


# ============================================================
# 6. Example usage
# ============================================================
user_id = "b80344d063b5ccb3212f76538f3d9e43d87dca9e"
print("Recommendations for user:")

recs = recommend_songs(user_id, n_recommendations=10)
print(recs)
