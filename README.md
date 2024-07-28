# Letterboxd Movie Recommender and Anaysis Website
Website Link: https://letterboxd-recommender.streamlit.app/
## Overview
This repository is home to a Streamlit web application that allows users to submit their personal Letterboxd user ratings and recieve personalized movie recommendations and preference analysis from those ratings

## Installation 
If you want to download the project to make your own, follow the instructions below. 

### Code used 
Ran in VsCode. Used Python==3.12.4

### Packages Used
Run 'conda env create -f environment.yml' to install necessary packages

## Data
### Source Data
All source data is from Kaggle. You can view the 'About' tab on the website to get linkes to the original Kaggle datasets. The original source data includes: genres.csv, languages.csv, movies.csv, posters.csv, releases.csv, TMDB_Dataset.csv. You can download all necessary data from the Data folder. 

### Data Preprocessing
All data preprocessing was done in the 'data_preprocessing_and_eda' Jupyter notebook. The movies_cleaned.csv file is a result of this data preprocessing. 

## Code Structure
- model_testing_and_creation: Jupyter notebook used primarily for model testing purposes and would not be at all necessary in any pulls from this repo
- app.py: The main script for the Streamlit app. 
- data_preprocessing_and_eda: Notebook for EDA and data cleaning/preprocessing
- Data Folder: Folder that holds all .csv files used for this project
- PersonalStuff: Files used for the 'About Me' tab in my version of the web app. Not needed for the functionality of anyone pulling

## Future Work
Something I may want to include in the future is the ability for users to create an account and save their model. They would then be allowed to add movies with a rating on the website and the model would update when they do this. I am also looking to add any new features that users are interesting in analysis wise. 

## Acknowledgements
Thanks to @asaniczka and @gsimonx37 on Kaggle for supplying the starting datasets. 
