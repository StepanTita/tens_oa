import streamlit as st

st.set_page_config(
    page_title="Train Test Split",
    page_icon="📊",
)

st.write("# Train Test Split! 📊")

st.sidebar.success("Select a page above.")

st.write('## Introduction')

st.write('We will use the even_train_test_split function to split the data into train and test sets. This function will split the data in such a way that each user has an equal number of ratings in both the train and test sets.')
st.write('We also add a function ratings_train_test_split that will make sure that all of the movies are present in both test and train (otherwise we won\'t be able to create a matrix).')

st.write('## Code')

st.code(
    '''
    def even_train_test_split(data, random_state=42):
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
    
        for user in data['userId'].unique():
            X = data[data['userId'] == user]
            X_train, X_test = train_test_split(X, test_size=0.5, random_state=random_state)
    
            train_df = pd.concat([train_df, X_train], axis=0, ignore_index=True)
            test_df = pd.concat([test_df, X_test], axis=0, ignore_index=True)
    
        return train_df, test_df
        
    def ratings_train_test_split(data, random_state=42):
        train_df, test_df = even_train_test_split(data, random_state=random_state)
    
        movies_ids = data['movieId'].unique()
    
        # make sure that all of the movies are present in both test and train (otherwise we won't be able to create a matrix)
        for movie in movies_ids:
            if movie not in train_df['movieId'].values:
                train_df = train_df.append(pd.Series({'userId': 1, 'movieId': movie, 'rating': pd.NA}), ignore_index=True)
            if movie not in test_df['movieId'].values:
                test_df = test_df.append(pd.Series({'userId': 1, 'movieId': movie, 'rating': pd.NA}), ignore_index=True)
        return train_df, test_df
    '''
)