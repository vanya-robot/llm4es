"""
Словари для перевода кодов в текст (рейтинги, жанры).
"""
from typing import Dict

# Рейтинг -> текстовая оценка
RATING_MAP: Dict[int, str] = {
    1: "terrible",
    2: "bad",
    3: "average",
    4: "good",
    5: "excellent",
}


def rating_to_text(rating: int) -> str:
    return RATING_MAP.get(rating, f"rating-{rating}")


# Жанр -> полное название (для ML-100K по сути identity, но расширяемо)
GENRE_MAP: Dict[str, str] = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Children's": "Children's",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Film-Noir": "Film Noir",
    "Horror": "Horror",
    "Musical": "Musical",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Sci-Fi": "Science Fiction",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western",
}


def map_genre(genre: str) -> str:
    return GENRE_MAP.get(genre.strip(), genre.strip())


def map_genres(genres_str: str) -> str:
    if not genres_str or genres_str == "unknown":
        return "unknown"
    parts = [map_genre(g) for g in genres_str.split(",")]
    return ", ".join(parts)
