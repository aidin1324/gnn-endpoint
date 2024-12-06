import requests
import json
import pandas as pd

# URL вашего эндпоинта
url = 'http://localhost:5000/recommend'

# Данные для отправки
payload = {
    "user_ratings": {
        1: 5,
        2: 4,
        3: 3,
        12: 5,
        11: 2,
        6: 4,
        15: 5,
        8: 3
    },
    "num_recommendations": 10,
    "diversity_factor": 0.1
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

movies_df = pd.read_csv("app/ml-latest-small/movies.csv")

print("Оценки пользователя:\n")
for movie_id, rating in payload["user_ratings"].items():
    print(f"Фильм: {movies_df[movies_df['movieId'] == movie_id]['title'].values[0]}, Оценка: {rating}")

print("\n")

if response.status_code == 200:
    recommended_movies = response.json()
    print("Рекомендованные фильмы:")
    for movie in recommended_movies:
        print(f"Название: {movie['title']}, Жанры: {movie['genres']}")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)