# app.py

import streamlit as st
from movie_recommender import recommend_movies_with_tags, movies_with_tags

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎥 Content-Based Movie Recommender")
st.markdown("Using genres and user tags to find similar movies")

# Dropdown for movie selection
movie_list = sorted(movies_with_tags["title"].dropna().unique())
selected_movie = st.selectbox("Select a movie:", movie_list)

# Recommend button
if st.button("Show Recommendations"):
    recommendations = recommend_movies_with_tags(selected_movie)
    if recommendations:
        st.success("Top 5 Recommended Movies:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.error("Sorry, no recommendations found.")
