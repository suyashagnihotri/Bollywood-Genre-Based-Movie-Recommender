# 🎬 Bollywood Genre-Based Movie Recommender (Suyash Agnihotri Project)

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk

# ----------------------------
# Step 1: Load Dataset (path-safe)
# ----------------------------

# Script directory
suyash_dir = os.path.dirname(os.path.abspath(__file__))
suyash_csv = os.path.join(suyash_dir, "movies_metadata.csv")

# Load dataset
suyash_df = pd.read_csv(suyash_csv, low_memory=False)

# Clean columns
suyash_df.columns = suyash_df.columns.str.strip().str.lower()
suyash_df = suyash_df[['title', 'genre']].dropna()
suyash_df.columns = ['title', 'genre_text']

# ----------------------------
# Step 2: TF-IDF + Cosine Similarity
# ----------------------------

suyash_vectorizer = TfidfVectorizer(stop_words='english')
suyash_matrix = suyash_vectorizer.fit_transform(suyash_df['genre_text'])

suyash_similarity = cosine_similarity(suyash_matrix, suyash_matrix)

suyash_index = pd.Series(suyash_df.index, index=suyash_df['title'].str.lower()).drop_duplicates()

def suyash_recommend(movie_title, top_n=5):
    """Return top N similar movies based on genre."""
    movie_title = movie_title.lower()
    if movie_title not in suyash_index:
        return []
    idx = suyash_index[movie_title]
    sim_scores = list(enumerate(suyash_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return suyash_df['title'].iloc[movie_indices].tolist()

# ----------------------------
# Step 3: Build GUI (CustomTkinter)
# ----------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

suyash_app = ctk.CTk()
suyash_app.geometry("520x470")
suyash_app.title("🎬 Bollywood Recommender — Suyash Agnihotri")

# Dropdown for movies
suyash_movies = sorted(suyash_df['title'].dropna().unique().tolist())
suyash_combo = ctk.CTkComboBox(suyash_app, values=suyash_movies, width=400)
suyash_combo.pack(pady=20)
suyash_combo.set("Select a movie")

# Output text box
suyash_output = ctk.CTkTextbox(suyash_app, width=460, height=250, font=("Arial", 12))
suyash_output.pack(pady=10)

# Button action
def suyash_show_recommendations():
    selected_movie = suyash_combo.get().strip()
    suyash_output.delete("1.0", "end")

    if not selected_movie or selected_movie == "Select a movie":
        suyash_output.insert("end", "⚠️ Please select a movie title.")
        return

    results = suyash_recommend(selected_movie)
    if results:
        suyash_output.insert("end", f"🎯 Recommendations for '{selected_movie}':\n\n")
        for i, rec in enumerate(results, 1):
            suyash_output.insert("end", f"{i}. {rec}\n")
    else:
        suyash_output.insert("end", "❌ Movie not found in dataset.")

# Recommend button
suyash_button = ctk.CTkButton(suyash_app, text="Recommend", command=suyash_show_recommendations)
suyash_button.pack(pady=10)

# Run GUI
suyash_app.mainloop()
