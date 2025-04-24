import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_and_prepare_data(n=15000):
    df = pd.read_csv("RAW_recipes.csv", usecols=["id", "name", "ingredients", "tags", "steps"])
    df = df.dropna(subset=['name'])  # Remove entries with missing names
    df = df.head(n)

    for col in ['ingredients', 'tags', 'steps']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join([i.lower().replace('-', ' ') for i in x]))
    df['tags_str'] = df['tags'].apply(lambda x: ' '.join([t.lower().replace('-', ' ') for t in x]))
    df['steps_str'] = df['steps'].apply(lambda x: ' '.join([s.lower().replace('-', ' ') for s in x]))
    df['combined_features'] = df['ingredients_str'] + ' ' + df['tags_str'] + ' ' + df['steps_str']
    return df

def build_model(df, max_features=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)

    return model, tfidf_matrix

def recommend_by_ingredient(ingredient_input, df, model, tfidf_matrix, top_n=5):
    ingredients = [ing.strip().lower() for ing in ingredient_input.split(',') if ing.strip()]

    # Recipe must include all input ingredients
    def recipe_matches(ing_list):
        ing_list_lower = [i.lower() for i in ing_list]
        return all(any(ing in i for i in ing_list_lower) for ing in ingredients)

    # Filter recipes that match all ingredients
    matches = df[df['ingredients'].apply(recipe_matches)]
    if matches.empty:
        return None

    # Use the first match as the seed
    recipe_index = matches.index[0]
    distances, indices = model.kneighbors(tfidf_matrix[recipe_index], n_neighbors=top_n * 3)

    # Refilter similar recipes to still match all input ingredients
    similar_recipes = df.iloc[indices[0][1:]]
    filtered = similar_recipes[similar_recipes['ingredients'].apply(recipe_matches)]

    return filtered[['name', 'ingredients', 'steps']].head(top_n)