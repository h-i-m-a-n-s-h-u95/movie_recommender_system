
import streamlit as st
import pickle
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor

# -------------------- Load Data --------------------
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))  # movie data
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))  # similarity matrix

# -------------------- TMDb Session with retries --------------------
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))


# -------------------- Poster Fetching --------------------
def fetch_poster_single(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6bbb28a637d35b071431a2e3548ddd3f&language=en-US"
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"


@st.cache_data
def fetch_posters_parallel(movie_ids):
    with ThreadPoolExecutor(max_workers=5) as executor:
        posters = list(executor.map(fetch_poster_single, movie_ids))
    return posters


# -------------------- Recommendation --------------------
def recommend(movie):
    indices = movies[movies['title'] == movie].index
    if len(indices) == 0:
        st.error("Movie not found!")
        return [], []
    index = indices[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = [movies.iloc[i[0]].title for i in distances[1:6]]
    movie_ids = [movies.iloc[i[0]].movie_id for i in distances[1:6]]
    recommended_movies_poster = fetch_posters_parallel(movie_ids)

    return recommended_movies, recommended_movies_poster


# -------------------- Streamlit UI --------------------
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Type or select a movie from the dropdown',
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    if names and posters:
        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.text(name)
                st.image(poster)


