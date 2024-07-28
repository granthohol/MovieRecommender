import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


@st.cache_data
def model_creation(user_ratings: pd.DataFrame):
    """
    Train a ridge regression model based on the users ratings and return the trained model

    Parameters:
    - user_ratings: Pandas dataframe of the users letterboxd ratings

    Returns:
    - Trained Ridge regression model
    """

    ########## Initialize Data ##############
    movies = pd.read_csv("Data/movies_cleaned.csv")
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
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=8, scoring='neg_mean_squared_error')
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
    movies = pd.read_csv("Data/movies_cleaned.csv")
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
    st.sidebar.title('Filters')
    st.sidebar.subheader('Use the filters to edit what types of movies come up as recommendations')
    # Filter for only english movies
    only_english = st.sidebar.checkbox("English Only", False)
    if only_english:
        movies_no_user = movies_no_user[movies_no_user['english'] == True]
    
    # filter for only kids movies
    only_kid = st.sidebar.checkbox("Kid Friendly", False)
    if only_kid:
        movies_no_user = movies_no_user[movies_no_user['adult'] == False]

    # Slider filter for minimum rating
    rating_min = st.sidebar.select_slider(
        'Letterboxd rating minimum',
        options=[round(x * 0.1,1) for x in range(0, 51)],
        value=(0.0)
    )
    movies_no_user = movies_no_user[movies_no_user['rating'] >= rating_min]

    # slider for maximum and minimum runtime
    runtime_slider = st.sidebar.select_slider(
        'Range for movie runtime', 
        options=list(range(45, 301)),
        value=(45, 300)
    )
    movies_no_user = movies_no_user[(movies_no_user['minute'] >= runtime_slider[0]) & (movies_no_user['minute'] <= runtime_slider[1])]

    # multiselect filter for genres
    genre_filter = st.sidebar.multiselect(
        "Must have genre(s)",
        ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 
         'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
    )
    movies_no_user = movies_no_user[movies_no_user[genre_filter].all(axis=1)]

    genre_none_filter = st.sidebar.multiselect(
        "Genres NOT to include",
        ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 
         'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
    )

    # Filter the DataFrame to exclude movies with the selected genres
    movies_no_user = movies_no_user[~movies_no_user[genre_none_filter].any(axis=1)]


    # Text input for minimum release year
    year_min = st.sidebar.text_input("Minimum Release Year", "1900")
    movies_no_user = movies_no_user[movies_no_user['date'] >= year_min]




    st.title('Top Movie Recommendations')


    # create df changed to look better for output
    output = movies_no_user.rename(columns={'name': 'Title', 'userRating': 'Predicted Rating', 'date':'Release Year', 'minute':'Runtime', 'rating':'Letterboxd Rating'})
    
    output = output.reset_index(drop=True)
    output.index = output.index + 1

    output['Predicted Rating'] = output['Predicted Rating'].round(2)

    # create new column genres that has a string literal of the one hot encoded features
    genre_cols = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
    def combine_genres(row):
        return ', '.join([genre for genre in genre_cols if row[genre] == True])
    
    output['Genres'] = output.apply(combine_genres, axis=1)

    output = output[['Title', 'Genres', 'Letterboxd Rating', 'Predicted Rating', 'Release Year', 'Runtime','adult', 'english', 'tagline', 'description']]
    

    # Output top 5 with image, description, etc

    for i in range(5):
        st.write('')
        col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

        with col1:
            st.subheader(movies_no_user.iat[i, 0])
            st.image(movies_no_user.iat[i, 6])
    
        with col2:
            st.write(output.iat[i, 1]) # genres
            st.write(movies_no_user.iat[i, 1]) # year
            st.write(f"Runtime: {movies_no_user.iat[i, 4]} minutes")
            st.write(f"Letterboxd Rating: {movies_no_user.iat[i, 5]}")
            st.write(movies_no_user.iat[i, 2]) # tagline
            st.write(movies_no_user.iat[i, 3]) # description


    # output full table
    st.write("")
    st.write("")
    st.header('Full Table')
    st.write('Expand to see all features')
    st.write('Use search button to search for a movie. You can also sort by column by clicking on the header.')
    st.dataframe(output)


def print_analysis(movies_no_user, user_ratings, model):
    """
    Prints visuals of the users personalized movie preferences and stats

    Parameters:
    - movies_no_user: the df of movies the user has not seen with their predicted rating
    - user_ratings: the df of movies the user has seen with their personal ratings
    - model: the Ridge regression model trained to predict user ratings
    
    Returns: 
    - None
    """

    ############### Feature Importance Output ########################
    coefficients = model.coef_

    feature_names = ['rating', 'adult', 'date', 'num_votes', 'minute', 'english',
                     'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                     'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    ######## Genre ################
    genre_feature_importance = feature_importance[feature_importance['Feature'].isin(['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 
            'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'])]

    st.header("Genre")
    st.write('Expand for a better view')
    # Initialize Matplotlib figure and axis
    fig, genre = plt.subplots(figsize=(24, 20))

    # Plotting on the axis
    genre.barh(genre_feature_importance['Feature'], genre_feature_importance['Coefficient'], color=['green' if coef > 0 else 'red' for coef in genre_feature_importance['Coefficient']])
    genre.set_xlabel('Importance')
    genre.set_title('Personal Genre Preferences', fontsize=36)
    genre.invert_yaxis()  # Highest importance at the top
    genre.tick_params(axis='y', labelsize=32)

    # Add legend for custom colors
    positive_bar = plt.Line2D([0, 1], [0, 0], color='green', linewidth=4, linestyle='-')
    negative_bar = plt.Line2D([0, 1], [0, 0], color='red', linewidth=4, linestyle='-')
    genre.legend([positive_bar, negative_bar], ['You tend to like this genre', 'You tend to not like this genre'], loc='lower right', fontsize=24)

    # Display the plot using st.pyplot()
    st.pyplot(fig)
    st.write('This graph tells you how much each genre, removing all other factors, effects your personal ratings. ' + 
             'The further to the right, the more you like that genre, and the further to the left, the more you dislike that genre, according to the model.')
    

    # genre freq plot

    movies = pd.read_csv("Data/movies_cleaned.csv")
    user_ratings = user_ratings.rename(columns={'Name' : 'name', 'Rating':'userRating', 'Year' : 'date'})
    user_ratings = user_ratings.drop(columns=['Date', 'Letterboxd URI'])
    movies_with_user = pd.merge(user_ratings, movies, how='inner', on=['name', 'date'])
    movies_with_user = movies_with_user.drop_duplicates(subset=['name', 'date'])
    
    import random

    colors = []
    genre_counts = movies_with_user.loc[:, 'Action':].drop(columns=['english']).sum()
    for _ in range(len(genre_counts)):
        colors.append('#%06x' % random.randint(0, 0xFFFFFF)) 

    genre_counts = genre_counts.sort_values(ascending=False)


    plt.figure(figsize=(24, 20))
    genre_counts.plot(kind='bar', color=colors)
    plt.title('What Genres Do You Watch the Most?', fontsize=36)
    plt.xlabel('Genre', fontsize=32)
    plt.ylabel('Frequency', fontsize=32)
    plt.xticks(rotation=45, ha='right', fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    st.pyplot(plt)

    st.write("Now, which genres are you watching the most? Does this align with your 'favorite' genres as seen above? If not, try to change up what you're watching.")
    


    ########## Runtime ##############
    other_feat_importance = feature_importance[feature_importance['Feature'].isin(['english', 'num_votes', 'minute', 'date', 'adult'])]
    other_feat_importance.loc[other_feat_importance['Feature'] == 'date', 'Feature'] = 'Release Year'
    other_feat_importance.loc[other_feat_importance['Feature'] == 'num_votes', 'Feature'] = 'Popularity'
    other_feat_importance.loc[other_feat_importance['Feature'] == 'minute', 'Feature'] = 'Runtime'
    other_feat_importance.loc[other_feat_importance['Feature'] == 'adult', 'Feature'] = 'Is Adult'
    other_feat_importance.loc[other_feat_importance['Feature'] == 'english', 'Feature'] = 'In English'

    st.write('')
    st.header("By Other Features")

    fig2, others = plt.subplots(figsize=(24,20))
    others.barh(other_feat_importance['Feature'], other_feat_importance['Coefficient'], color=['green' if coef > 0 else 'red' for coef in other_feat_importance['Coefficient']])
    others.set_xlabel('Importance')
    others.set_title('Personal Features Preference', fontsize=36)
    others.invert_yaxis()
    others.tick_params(axis='y', labelsize=32)

    positive_bar = plt.Line2D([0, 1], [0, 0], color='green', linewidth=4, linestyle='-')
    negative_bar = plt.Line2D([0, 1], [0, 0], color='red', linewidth=4, linestyle='-')
    others.legend([positive_bar, negative_bar], ['You tend to like when this feature is greater or true', 'You tend to like when this feature is less or false'], loc='lower right', fontsize=24)

    st.pyplot(fig2)
    st.write('Same idea here! If the bar is to the right, you like when that feature is a greater value or true (later release year, higher popularity, in english, etc).' +
             ' If the bar is to the left, you like when that feature is a lower value or false (Is NOT in English, shorter runtime, etc).')
    

    ########## Ratings Analysis ###############
    import plotly.express as px

    st.write('')
    st.title('Your Ratings vs Letterboxd Ratings')


    
    
    # Create a Plotly figure
    fig3 = px.scatter(movies_with_user, x='rating', y='userRating', trendline='ols',
                 title='User Ratings vs Letterboxd Ratings',
                 hover_name='name', hover_data={'rating': True, 'userRating': True})

    fig3.update_layout(
        xaxis_title='Letterboxd Rating',
        yaxis_title='User Rating',
        hoverlabel=dict(
            bgcolor="red",  # Background color of tooltip
            font_size=12,     # Font size of tooltip text
            font_family="Arial"
        ),
        height=600,
        width=800
    )

    st.plotly_chart(fig3) 
    st.write('Here you can see the correlation between your ratings and letterboxd ratings. Expand for a better view and hover over dots to see which movie it is.')
    st.write('')

    # distribution plot
    plt.figure(figsize=(24, 20))

    sns.kdeplot(movies_with_user['userRating'], color='blue', fill=True, linewidth=2 )
    sns.kdeplot(movies_with_user['rating'], color='red', fill=True, linewidth=2)

    plt.title('Distribution of Your Ratings vs Letterboxd Ratings', fontsize=36)
    plt.xlabel('Rating', fontsize=32)
    plt.ylabel('Frequency', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(['Your Ratings', 'Letterboxd Ratings'], fontsize=24)

    st.pyplot(plt)

    st.write("Here is a distribution of your ratings vs Letterboxd ratings for the movies you watched. Do you tend to over rate or under rate? "+
             "Do you have more variability or less variability in your ratings?")



    # Top 5 movies liked more and liked less

    movies_with_user['Difference'] = movies_with_user['userRating'] - movies_with_user['rating']
    temp = movies_with_user.rename(columns={'userRating': 'Personal Rating', 'rating':'Letterboxd Rating', 'name':'Movie'})
    movies_liked_more = temp.sort_values(by=['Difference'], ascending=False)[['Movie', 'Personal Rating', 'Letterboxd Rating', 'Difference']]
    movies_liked_less = temp.sort_values(by=['Difference'], ascending=True)[['Movie', 'Personal Rating', 'Letterboxd Rating', 'Difference']]

    st.header('Top 5 Movies You Liked More than Letterboxd')
    movies_liked_more = movies_liked_more.reset_index(drop=True)
    movies_liked_more.index = movies_liked_more.index + 1
    st.dataframe(movies_liked_more.head())
    st.write('')

    st.header('Top 5 Movies You Liked Less than Letterboxd')
    movies_liked_less = movies_liked_less.reset_index(drop=True)
    movies_liked_less.index = movies_liked_less.index + 1
    st.dataframe(movies_liked_less.head())

    st.write('Putting it all together, here are the top 5 movies you liked more/less than the Letterboxd consensus.')


@st.cache_data
def print_howto():
        st.title("How To Use the Website")
        st.write('On this page, I will walk you through how to use the website. It is fairly simple, but you need to be sure you are submitting' +
                 ' the correct file, so pay attention!')
        st.header('Step One: Letterboxd')
        st.write('First things first, you are going to need a Letterboxd account. If you have one, great! You can move on. '+
                 'If not, you should really consider creating one and coming back after you have logged some movies. It is a great app, '+
                 f'especially if you are a frequent movie watcher. You can create an account here: {'https://letterboxd.com/welcome/'}')
        st.write('If you want to test out the website and its functionality without an account, you can use my ratings: ')

        # test data for user
        file_path = 'Data/user_test_ratings.csv'
        data = pd.read_csv(file_path)

        import base64
        def download_csv():
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV file
            href = f'<a href="data:file/csv;base64,{b64}" download="movie_test_ratings.csv">Download CSV file</a>'
            return href

        # Display the download button
        st.markdown(download_csv(), unsafe_allow_html=True)
        st.write('Now just use that downloaded file to submit on the home page.')


        st.header('Step Two: Downloading your ratings - Laptop or Desktop')
        st.write('Please note: This process will most likely be easier from a laptop or desktop.')
        st.markdown("1. Navigate to [Letterboxd](https://letterboxd.com/welcome/)")
        st.write('2. Sign in to your account')
        st.write('3. Navigate to Settings > Data > Export Your Data')
        st.write("4. After that, you should get a downloaded zip file on your computer starting with 'letterboxd'")
        st.write("5. Extract this file to a location of your choosing, and now you are ready to go!")

        st.header('Step Two: Downloading your ratings - IOS')
        st.write('As mentioned above, I encourage you to try this out on laptop/desktop')
        st.write('1. Open your Letterboxd app')
        st.write('2. Navigate to Account (far right icon) > Settings (top left) > Advanced Settings > Account Data (At bottom) > Export Your Data')
        st.write("3. You should recieve a .zip download. Within this zip is a file called 'ratings.csv'. If you see this, you are ready to go!")

        st.header('Step Three: Submitting your ratings')
        st.write("Navigate back to the website and click on the 'Browse files' button." +
                  " Submit the file 'ratings.csv' from your downloaded folder and watch the magic happen!")
        st.write("Important: Make sure you are submitting 'ratings.csv' and nothing else!")    

def print_about():
        st.header("The Data")
        st.markdown("The data for this website was obtained from two different Kaggle datasets." + 
                 " Big thanks to [Simon Garanin](https://www.kaggle.com/gsimonx37) and [Asanickza](https://www.kaggle.com/asaniczka) for supplying these.")
        st.markdown("1. [Letterboxd data](https://www.kaggle.com/datasets/gsimonx37/letterboxd)")
        st.markdown("2. [Data for movie popularity](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)")

        st.write('')

        st.header('The Model')
        st.subheader('Type')
        st.write('The model used to make ratings predictions is a Ridge Regression model that gets trained on the user data once submitted.')
        st.write('I chose a Ridge Regression model for a variety of reasons')
        st.write(' 1. Ridge Regression performed well compared to other models on test data from my ratings and some of my friends.')
        st.write('2. It is  a fast and interpretable model which helps a lot with what I am trying to accomplish -- training the model in live time and ' +
                 'outputting feature coefficients to the user.')
        st.write('3. Regularizaiton impact: ' +
                 'The penalization of coefficients to prevent overfitting by regulating the loss function is crucial here because I have a feature (Letterboxd Rating) '+
                 'that kind of dominates most of the other features and may lead to overfitting in a lot of other models.')
        st.subheader('Parameters')
        st.write('The models parameters are determined, again, in live time after user submission. I use a Grid Search to determine the best parameters for the particular user.')
        st.markdown("You can check out the code for this website in more detail on my [github](https://github.com/granthohol/MovieRecommender.com)")

        st.header('Bugs and Suggestions')
        st.write('Any bugs on the website or general suggestions you have can be emailed to ghohol@wisc.edu.'+
                 ' I appreciate any and all feedback! I will always be looking to add additional features.' +
                 ' You can also pull the code from the github linked above to add anything you want yourself.')
        
        st.header('Patch Notes')   
        st.write('To come')

@st.cache_data
def print_me():
        st.title("About The Creator")
        st.header("Grant Hohol")
        st.markdown("Hi, I'm Grant. I am a sophomore at the University of Wisconsin-Madison, where I study Computer Sciences and Statistics, both of which helped me create this website, although I am also largely self taught. " +
                     "I love the challenges and rewards of unraveling complex data through coding and finding interesting insights. Away from the keyboard, " +
                     "I'm a sports junkie (somewhere I like to deploy my data skills, as you can see on my resume/github), an avid reader (check out my [storygraph](https://app.thestorygraph.com/profile/granthohol55)), " +
                     "and a staunch pursuer of personal fitness (follow me on [Strava](https://www.strava.com/athletes/122667425)).")
        st.write("Here are some more places you can check out my work, my resume, or get in contact. I'm looking for internships or any cool projects I can help out on!")

            # Function to convert a PDF file to a base64 string
        import base64
        def get_base64_of_bin_file(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        pdf_base64 = get_base64_of_bin_file('PersonalStuff/Resume - Grant Hohol.pdf')

        st.markdown(f'<a href="data:application/pdf;base64,{pdf_base64}" download="resume.pdf">~ Download resume</a>', unsafe_allow_html=True)
        st.markdown("~ Github: [@granthohol](https://github.com/granthohol/)")
        st.markdown("~ [LinkedIn](https://www.linkedin.com/in/grant-hohol-08520b291/)")
        st.markdown("~ X (Twitter): [@granthohol55](https://x.com/granthohol55)")
        st.write("~ Email: ghohol@wisc.edu")
        st.write("~ Phone: 920-370-2380")


def main():



    home, howto, about, me = st.tabs(['Home', 'How To', 'About', 'About Me'])

    with home:
            # Set app title
        st.title("Letterboxd Movie Recommender")
        st.write(
            "A machine learning app to predict personal movie ratings for Letterboxd users and provide recommendations. Also provides analysis of the users movie ratings and characteristic preferences."
        )
        # Get ratings.csv file from user
        csv_file = st.file_uploader(
            "Upload your ratings.csv file from Letterboxd to get recommendations and insights. Note: It may take a second.", type="csv")

        # Read into df, throw error message if it does not work
        if csv_file is not None: 
            try: 
                user_ratings = pd.read_csv(csv_file)
                test = user_ratings.drop(columns=["Letterboxd URI"])
            except Exception as e:
                st.write("Please try again. Enter a valid .csv file.")
            else: 
                # Do everything else pretty much
                model = model_creation(user_ratings)
                df_preds = predict(model, user_ratings)

                preds, analysis = st.tabs(['Recommendations', 'Analysis'])
                with preds:
                    print_recs(df_preds)
                with analysis:
                    st.title("Personalized Preference Analysis")
                    print_analysis(df_preds, user_ratings, model)
    
    with howto:
        print_howto()

    with about:
        print_about()

    with me:
        print_me()


        

           


if __name__ == "__main__":
    main()    

