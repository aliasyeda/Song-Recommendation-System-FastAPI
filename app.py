from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create FastAPI app

app = FastAPI(title="Song Recommendation System API")


# Load and prepare dataset

df = pd.read_csv("spotify_millsongdata.csv")

# Remove songs with empty lyrics
df = df[df["text"].notnull()]

# Reduce dataset size for faster execution
df = df.sample(5000, random_state=42).reset_index(drop=True)

# Combine features into one text column
df["combined_features"] = (
    df["artist"] + " " +
    df["song"] + " " +
    df["text"]
)


# TF-IDF Vectorization

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df["combined_features"])


# Cosine Similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation function

def recommend_songs(song_name: str, top_n: int = 5):
    if song_name not in df["song"].values:
        return []

    # Get index of the song
    index = df[df["song"] == song_name].index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(cosine_sim[index]))

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get top recommendations 
    song_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    # Prepare result
    recommendations = []
    for i in song_indices:
        recommendations.append({
            "song": df.iloc[i]["song"],
            "artist": df.iloc[i]["artist"]
        })

    return recommendations


# API Endpoint

@app.get("/recommend")
def get_recommendations(song: str, top_n: int = 5):
    recommendations = recommend_songs(song, top_n)

    if not recommendations:
        return {"error": "Song not found in dataset"}

    return {
        "input_song": song,
        "recommendations": recommendations
    }
