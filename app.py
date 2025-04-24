import streamlit as st
from recommender_backend import load_and_prepare_data, build_model, recommend_by_ingredient

st.set_page_config(page_title="AI Meal Recommender", page_icon="🍽️")

@st.cache_resource
def init_system():
    df = load_and_prepare_data()
    model, tfidf_matrix = build_model(df)
    return df, model, tfidf_matrix

df, model, tfidf_matrix = init_system()

# UI Layout
st.title("🍽️ Ingredient-Based Recipe Recommendation")
st.write("Enter an ingredient, and we'll recommend recipes that include it!")

user_input = st.text_input("Enter ingredient(s):", placeholder="e.g., chicken, garlic, rice")

if user_input:
    results = recommend_by_ingredient(user_input, df, model, tfidf_matrix)

    if results is not None:
        st.success(f"Here are recipes that include **{user_input.strip()}**:")
        for i, row in results.iterrows():
            search_url = f"https://www.google.com/search?q={row['name'].replace(' ', '+')}+recipe+site:food.com"
            with st.expander(f"🍲 [{row['name']}]({search_url})", expanded=False):
                st.markdown(f'<a href="{search_url}" target="_blank">🔗 View full recipe online</a>', unsafe_allow_html=True)
                st.markdown(f"**Ingredients:** {', '.join(row['ingredients'])}")
                st.markdown("**Full Steps:**")
                for step in row['steps']:
                    st.markdown(f"- {step}")
    else:
        st.warning("❌ No recipes found with these ingredients. Try entering something different.")