# AI Ingredient-Based Recipe Recommender

This project is an AI-powered meal recommendation system designed to help users discover recipes based on ingredients they already have. Users can input one or more ingredients, and the system returns matching recipes that include those ingredients, complete with instructions and external links.

## Features
- Accepts one or more user-provided ingredients (comma-separated)
- Returns recipes that include all specified ingredients
- Displays ingredients, preparation steps, and a link to a full recipe online
- Web-based interface built using Streamlit

## Project Overview

1. Loads a dataset of over 2,000 recipes from `RAW_recipes.csv`
2. Cleans and tokenizes ingredients, tags, and steps
3. Vectorizes recipe content using TF-IDF
4. Filters recipes that contain all user-input ingredients
5. Ranks results by content similarity using Nearest Neighbors

## Technologies Used

- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Python (Pandas, Scikit-learn)
- **Model**: TF-IDF Vectorizer with Nearest Neighbors for similarity matching
- **Data**: Food.com recipe dataset (Kaggle)

