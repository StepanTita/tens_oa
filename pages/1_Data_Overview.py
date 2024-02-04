import streamlit as st

st.set_page_config(
    page_title="Data overview",
    page_icon="📊",
)

st.write("# Movie recommendation Data overview! 📊")

st.sidebar.success("Select a page above.")


def prepare_data(ratings, movies, movies_metadata):
    return ratings, movies, movies_metadata


@st.cache_data
def load_data():
    import pandas as pd

    ratings = pd.read_csv('data/ratings.csv')

    movies = pd.read_csv('data/movies_mapped.csv')

    movies_metadata = pd.read_csv('data/movies_metadata.csv')

    return prepare_data(ratings, movies, movies_metadata)


def show_data_overview(ratings, movies, movies_metadata):
    import pandas as pd
    # most viewed movies

    st.write("## Most viewed movies")

    st.bar_chart(
        ratings.join(
            movies[['movieId', 'title']].set_index('movieId'),
            how='inner', on='movieId'
        )['title'].value_counts().sort_values(ascending=False).head(10)
    )

    st.write("## Ratings distribution")
    st.bar_chart(ratings['rating'].value_counts())

    st.write("## Genres")
    st.bar_chart(movies['genres'].apply(lambda x: x.split('|')).explode().value_counts())

    # combine budgets into groups
    st.write("## Budgets")

    movies_metadata['budget'] = pd.to_numeric(movies_metadata['budget'], errors='coerce')
    bins = [0, 1e6, 5e6, 1e7, 1e8, 1e9]

    # Define the labels for the bins
    labels = ['<1M', '1M-5M', '5M-10M', '10M-100M', '100M-1B']

    # Create a new column 'budget_range' by categorizing 'budget' column
    movies_metadata['budget_range'] = pd.cut(movies_metadata['budget'], bins=bins, labels=labels)
    st.bar_chart(movies_metadata['budget_range'].value_counts())

    st.write("## Movies")
    st.write(movies.head())

    st.write("## Movies Metadata")
    st.write(movies_metadata.head())

    st.write("## Ratings")
    st.write(ratings.head())

    # st.write("## Production companies")
    # st.bar_chart(
    #     movies_metadata['production_companies'].apply(lambda x: [o['name'] for o in x]).explode().value_counts())



ratings, movies, movies_metadata = load_data()

show_data_overview(ratings, movies, movies_metadata)
