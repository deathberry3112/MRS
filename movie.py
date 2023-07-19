import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('E:\jprograms\movies.csv')

overview = movies['overview'].tolist()


vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(overview)


similarity = cosine_similarity(vectors)


def recommend_movies(title, similarity=similarity):
  
  idx = movies[movies['title'] == title].index[0]

  
  sim_scores = list(enumerate(similarity[idx]))

  
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  sim_scores = sim_scores[1:6]


  movie_ids = [i[0] for i in sim_scores]


  return movies['title'].iloc[movie_ids]

st.title('Movie Recommender System')


title = st.text_input('Enter the title of a movie:')


if st.button('Get Recommendations'):

  recommendations = recommend_movies(title)


  st.write('Recommendations:')
  st.write(recommendations)