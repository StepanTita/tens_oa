import streamlit as st

st.set_page_config(
    page_title="Features enhancement",
    page_icon="📊",
)

st.markdown("""
# Recommendation Abstract

## Rationale

* We are using external data to enhance the recommendation system. We are adding movie metadata to the existing data.

* We also use ratings and separate Matrix Factorization model to predict ratings, and use it as a feature for the AdaBoost model.

* We want to predict both based on the similarities of the movies and the users.

* For that we combine 2 models and use the output of the first model as a feature for the second model.

* To do that we use a function "rank_data" that creates the required set of features for the models prediction
""")

st.code(
    """
    def rank_data(data, users_sim, svd, top_n=5):
        for i in range(5):
            data[f'su_{i}'] = 0.0
        data['svd_rating'] = 0.0
    
        for i, (userId, movieId) in enumerate(data[['userId', 'movieId']].values):
            data.iat[i, data.columns.get_loc('svd_rating')] = svd.predict(int(userId), int(movieId)).est
            similar_ratings = get_similar_users_ratings(userId, movieId, users_sim, svd, top_n)
    
            for uid, r in enumerate(similar_ratings):
                data.iat[i, data.columns.get_loc(f'su_{uid}')] = r
        return data
    """
)

st.markdown("""
* We use AdaBoost as an example of a simple to use and efficient model for the recommendation system.
* We could also use other models like XGBoost, LightGBM, or CatBoost in the real-world scenario.
""")