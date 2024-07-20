import streamlit as st 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



def model_creation(user_ratings: pd.DataFrame):
    """
    Train a ridge regression model based on the users ratings and return the trained model

    Parameters:
    - user_ratings: Pandas dataframe of the users letterboxd ratings

    Returns:
    - Trained Ridge regression model
    """

    ########## Initialize Data ##############
    movies = pd.read_csv("D:/MovieRecommender/Data/movies_cleaned.csv")
    user_ratings = user_ratings.rename(columns={'Name' : 'name', 'Rating':'userRating', 'Year' : 'date'})
    user_ratings = user_ratings.drop(columns=['Date', 'Letterboxd URI'])
    movies_with_user = pd.merge(user_ratings, movies, how='inner', on=['name', 'date'])
    movies_with_user = movies_with_user.drop_duplicates(subset=['name', 'date'])


    ########### Model Creation ################
    # Train test split
    features = movies_with_user.drop(columns=['userRating', 'name', 'description', 'poster', 'tagline', 'id'])
    target = movies_with_user['userRating']


    # Train model
    # Define the parameter grid
    param_grid = {
        'alpha' : [0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept' : [True, False],
        'max_iter' : [None, 100, 500, 1000],
        'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
        'random_state' : [42]

    }

    ridge = Ridge()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(features, target)

    # Train the model with the best parameters
    best_ridge = grid_search.best_estimator_  
    best_ridge.fit(features, target)

    return best_ridge



def predict(model, user_ratings):
    """
    Method to create a new df with predictions from the model on movies the user has not seen

    Parameters:
    - model: Trained Ridge regression model
    - user_ratings: 

    Returns:
    - df with predictions added
    """

    ########### Initialize Data #######################
    movies = pd.read_csv("D:/MovieRecommender/Data/movies_cleaned.csv")
    user_ratings = user_ratings.rename(columns={'Name' : 'name', 'Rating':'userRating', 'Year' : 'date'})
    user_ratings = user_ratings.drop(columns=['Date', 'Letterboxd URI'])
    merged_df = pd.merge(movies, user_ratings, how='outer', on=['name', 'date'])
    movies_no_user = merged_df[merged_df['userRating'].isnull()].drop(columns='userRating')

    ############ Predictions ##########################
    # Features to predict on 
    movies_no_user_feat = movies_no_user.drop(columns=['name', 'description', 'poster', 'tagline'])
    movies_no_user_feat = movies_no_user_feat.sort_values(by=['id'], ascending=True)

    # Create new df with the predicted ratings
    ratings = pd.DataFrame(columns=['id', 'userRating'])
    ratings['id'] = movies_no_user['id'] 
    ratings = ratings.sort_values(by='id')
    ratings['userRating'] = model.predict(movies_no_user_feat.sort_values(by='id').drop(columns='id'))

    # merge predicted ratings df with its corresponding movie
    movies_no_user = pd.merge(movies_no_user, ratings, how='inner', on=['id'])

    # Sort df with predicted ratings by that rating
    movies_no_user = movies_no_user.sort_values(by=['userRating'], ascending=False).drop(columns='id')


    # Set any predictions greater than 5 to 5
    movies_no_user.loc[movies_no_user['userRating'] > 5, 'userRating'] = 5
    movies_no_user['date'] = movies_no_user['date'].astype(str).str.replace(",", "").str.replace(".0","")

    return movies_no_user



def print_recs(movies_no_user: pd.DataFrame):
    """
    Method to output the top movie recommendations in the app

    Parameters: 
    - movies_no_user: The df with predictions for every movie the user has not seen, sorted by predicted rating

    Returns: 
    - None

    """
    ########### Output to app ##################

    # Sidebar
    only_english = st.sidebar.checkbox("English Only", False)
    if only_english:
        movies_no_user = movies_no_user[movies_no_user['english'] == True]

    st.header('Top Movie Recommendations')

    # Output top 5 with image, description, etc

    for i in range(5):
        col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

        with col1:
            st.subheader(movies_no_user.iat[i, 0])
            st.image(movies_no_user.iat[i, 6])
    
        with col2:
            st.write(movies_no_user.iat[i, 1])
            st.write(f"Runtime: {movies_no_user.iat[i, 4]} minutes")
            st.write(f"Letterboxd Rating: {movies_no_user.iat[i, 5]}")
            st.write(movies_no_user.iat[i, 2])
            st.write(movies_no_user.iat[i, 3])


    st.write(movies_no_user.reset_index)



def main():

    # Set app title
    st.title("Movie Recommender and Analysis")
    st.write(
        "A machine learning app to predict personal movie ratings for Letterboxd users. Also provides analysis of the users movie characteristic preferences."
    )

    # Get ratings.csv file from user
    csv_file = st.file_uploader(
        "Upload your ratings.csv file from Letterboxd to get recommendations and insights.\nRemember: The more movies logged the better!\nNote: It may take a second.", type="csv")

    # Read into df, throw error message if it does not work
    if csv_file is not None: 
        try: 
           user_ratings = pd.read_csv(csv_file)
        except Exception as e:
            st.write("Error:", e)
            st.write("Please try again. Enter a valid .csv file.")
        else: 
           # Do everything else pretty much
           model = model_creation(user_ratings)
           df_preds = predict(model, user_ratings)
           print_recs(df_preds)
           


if __name__ == "__main__":
    main()    

