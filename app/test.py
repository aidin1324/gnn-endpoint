import requests
import json

# URL вашего эндпоинта
url = 'http://localhost:5000/recommend'

# Данные для отправки
payload = {
    "user_ratings": {
        0: 5,
        1: 4,
        2: 3,
        3: 5,
        4: 2,
        5: 4,
        6: 5,
        7: 3
    },
    "num_recommendations": 20,
    "diversity_factor": 0.3
}

# Заголовки запроса
headers = {
    "Content-Type": "application/json"
}

# Отправка POST-запроса
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Проверка статуса ответа и вывод результата
if response.status_code == 200:
    recommended_movies = response.json()
    print("Рекомендованные фильмы:")
    for movie in recommended_movies:
        print(f"Название: {movie['title']}, Жанры: {movie['genres']}")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)