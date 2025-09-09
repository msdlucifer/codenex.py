import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title','genres','keywords','overview']].fillna('')

movies['description'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']

cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(movies['description'])

similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in movies['title'].values:
        print("Movie not found")
        return
    idx = movies[movies['title']==movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    for i,score in scores:
        print(movies.iloc[i].title, "(", round(score,3), ")")

movie_name = input("Enter a movie title: ")
recommend(movie_name)
