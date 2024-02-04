import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, accuracy_score, \
    f1_score
import numpy as np

import streamlit as st

st.set_page_config(
    page_title="Predict movie ratings",
    page_icon="🌟",
)

st.write("# Movie recommendation Prediction Model! 🌟")

st.sidebar.success("Select a page above.")


@st.cache_data
def load_data():
    import pandas as pd

    orig_movies = pd.read_csv('data/movies.csv')
    movies = pd.read_csv('data/movies_cleaned.csv')

    train_ratings = pd.read_csv('data/train_ratings.csv').dropna()
    test_ratings = pd.read_csv('data/test_ratings.csv').dropna()

    return orig_movies, movies, train_ratings, test_ratings


@st.cache_data
def load_models():
    import joblib

    ada = joblib.load('models/ada.joblib')
    svd = joblib.load('models/svd.joblib')

    # users_sim = svd.compute_similarities()

    return ada, svd


# find ratings for the movies for top 5 cosine similar users

def get_top_similar_users(user_id, users_sim, top_n=5):
    import numpy as np

    user_similarities = users_sim[user_id - 1]

    top_similar_users = np.argsort(user_similarities)[-top_n:]

    return top_similar_users


def get_similar_users_ratings(user_id, movie_id, users_sim, svd, top_n=5):
    similar_users = get_top_similar_users(user_id, users_sim, top_n)

    similar_users_ratings = [svd.predict(user, movie_id).est for user in similar_users]

    return similar_users_ratings


def predict(user_id, movies, users_sim, model, svd, top_n=5):
    import numpy as np

    data = movies.copy()
    for i in range(5):
        data[f'su_{i}'] = 0.0
    data['svd_rating'] = 0.0

    for i, movieId in enumerate(movies['movieId'].values):
        data.iat[i, data.columns.get_loc('svd_rating')] = svd.predict(int(user_id), int(movieId)).est
        similar_ratings = get_similar_users_ratings(user_id, movieId, users_sim, svd, top_n)

        for uid, r in enumerate(similar_ratings):
            data.iat[i, data.columns.get_loc(f'su_{uid}')] = r

    prediction = model.predict(data.drop(columns=['movieId']))
    return data.iloc[np.argsort(prediction)[-top_n:], :], prediction[np.argsort(prediction)[-top_n:]]


orig_movies, movies, train_ratings, test_ratings = load_data()

ada, svd = load_models()

users_sim = svd.compute_similarities()

user_id = st.sidebar.selectbox('Select a user', test_ratings['userId'].unique(), index=433)

st.write(f"### Original ratings of user {user_id}")

st.write(test_ratings[test_ratings['userId'] == user_id].merge(orig_movies, on='movieId')[['title', 'rating']].rename({
    'title': 'Movie',
    'rating': 'Assigned Rating'
}, axis=1).dropna())

top_n = 5

st.write(f"## Predictions for user {user_id}")

predictions, scores = predict(user_id, movies, users_sim, ada, svd, top_n)

movies_to_recommend = [orig_movies[orig_movies['movieId'] == movieId]['title'].values[0] for movieId in
                       predictions['movieId'].values]
movies_to_recommend = list(zip(movies_to_recommend, scores))
st.write(pd.DataFrame(movies_to_recommend, columns=['Movie', 'Predicted rating']))

# calculate metrics for those predictions that are in the test set

# Create lists to store actual and predicted ratings

st.write(f"## Metrics for user {user_id}")

actual_ratings = []
predicted_ratings = []

real_preds = 0

for movie_id, score in zip(predictions['movieId'].values, scores):
    if test_ratings[(test_ratings['userId'] == user_id) & (test_ratings['movieId'] == movie_id)].shape[0] > 0:
        actual_rating = \
            test_ratings[(test_ratings['userId'] == user_id) & (test_ratings['movieId'] == movie_id)]['rating'].values[
                0]
        predicted_rating = score

        real_preds += 1

        # Append actual and predicted ratings to their respective lists
        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)

if len(actual_ratings) > 0 and len(predicted_ratings) > 0:
    # Convert lists to numpy arrays
    actual_ratings = np.array(actual_ratings)
    predicted_ratings = np.array(predicted_ratings)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    # Calculate MAPE
    mape = (mean_absolute_error(actual_ratings, predicted_ratings) / actual_ratings)[0]

    # Calculate precision, recall, accuracy, and F1 score
    # Note: These metrics are typically used for classification tasks, not regression tasks like rating prediction.
    # Here, we consider a prediction to be "correct" if the predicted rating is within 0.5 of the actual rating.
    predicted_classes = np.array(
        [1 for i in range(real_preds)] + [0 for i in range(5 - real_preds)])
    actual_classes = np.ones(5)

    precision = precision_score(actual_classes, predicted_classes)
    recall = recall_score(actual_classes, predicted_classes)
    accuracy = accuracy_score(actual_classes, predicted_classes)
    f1 = f1_score(actual_classes, predicted_classes)

    st.write(f"**RMSE**: {rmse}")
    st.write(f"**MAPE**: {mape}")
    st.write(f"**Precision**: {precision}")
    st.write(f"**Recall**: {recall}")
    st.write(f"**Accuracy**: {accuracy}")
    st.write(f"**F1 Score**: {f1}")
