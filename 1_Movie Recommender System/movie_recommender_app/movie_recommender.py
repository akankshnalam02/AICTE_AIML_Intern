# movie_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# STEP 1: Load full dataset using absolute paths
movies = pd.read_csv("C:/Users/akanksh_02/AICTE_AIML_Intern/1_Movie Recommender System/movie_lens_data/movie.csv")            # movieId, title, genres
genome_tags = pd.read_csv("C:/Users/akanksh_02/AICTE_AIML_Intern/1_Movie Recommender System/movie_lens_data/genome_tags.csv")  # tagId, tag
genome_scores = pd.read_csv("C:/Users/akanksh_02/AICTE_AIML_Intern/1_Movie Recommender System/movie_lens_data/genome_scores.csv")  # movieId, tagId, relevance


# Merge genome scores with tags
genome_data = pd.merge(genome_scores, genome_tags, on="tagId")

# Select top 15 relevant tags per movie
top_tags_per_movie = genome_data.sort_values(["movieId", "relevance"], ascending=[True, False])\
                                .groupby("movieId").head(15)

# Combine tags into one string per movie
movie_tags = top_tags_per_movie.groupby("movieId")["tag"].apply(lambda tags: " ".join(tags)).reset_index()

# Merge movie tags with movie metadata
movies_with_tags = pd.merge(movies, movie_tags, on="movieId", how="left")
movies_with_tags["tag"] = movies_with_tags["tag"].fillna("")
movies_with_tags["genres"] = movies_with_tags["genres"].fillna("")

# Create combined profile (genres + tags)
movies_with_tags["profile"] = movies_with_tags["genres"] + " " + movies_with_tags["tag"]

# TF-IDF vectorizer on profile
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix_tags = tfidf_vectorizer.fit_transform(movies_with_tags["profile"])

# Compute cosine similarity
cosine_sim_tags = linear_kernel(tfidf_matrix_tags, tfidf_matrix_tags)

# Build reverse index: title → index
indices_tags = pd.Series(movies_with_tags.index, index=movies_with_tags["title"]).drop_duplicates()

# Recommendation function
def recommend_movies_with_tags(title, cosine_sim=cosine_sim_tags):
    idx = indices_tags.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_with_tags["title"].iloc[movie_indices].tolist()
