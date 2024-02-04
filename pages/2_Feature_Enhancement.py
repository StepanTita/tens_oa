# create a page explaining which features are enhanced in the new version

import streamlit as st

st.set_page_config(
    page_title="Features enhancement",
    page_icon="📊",
)

def prepare_data(ratings, movies, movies_metadata):
    return ratings, movies, movies_metadata

@st.cache_data
def load_data():
    import pandas as pd

    ratings = pd.read_csv('data/ratings.csv')

    movies = pd.read_csv('data/movies_mapped.csv')

    movies_metadata = pd.read_csv('data/movies_metadata.csv')

    return prepare_data(ratings, movies, movies_metadata)

ratings, movies, movies_metadata = load_data()

st.write("# Features enhancement! 📊")

st.sidebar.success("Select a page above.")

st.write('## External Data')

st.write('Since the description did not prohibit using external data we would add movie metadata.')

st.write('## New features')

st.write(movies_metadata[['title', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'production_companies']])

st.write('## Features transformation')

st.markdown("""
* We split the genres into separate boolean columns
* We split production companies into separate boolean (top 50) columns, the rest are considered as "other"
* We use ratings and separate Matrix Factorization model to predict ratings, and use it as a feature for the AdaBoost model
* We also find top 5 most similar users based on the cosine similarity of the factorized matrix, and add their ratings for the respective movie as features
""")