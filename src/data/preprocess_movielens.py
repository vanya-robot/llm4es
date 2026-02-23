"""
Предобработка ML-100K: сборка последовательностей событий по пользователям.

Каждое событие: {title, year, genres, rating, timestamp}
Плюс загрузка демографии (age, gender) для downstream-задач.
"""
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd

from src.utils.logging import get_logger


def load_movies(ml_dir: str) -> Dict[int, dict]:
    """Загрузить метаданные фильмов из u.item."""
    genre_names = []
    genre_path = os.path.join(ml_dir, "u.genre")
    if os.path.isfile(genre_path):
        with open(genre_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2 and parts[0]:
                    genre_names.append(parts[0])

    movies = {}
    item_path = os.path.join(ml_dir, "u.item")
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            movie_id = int(parts[0])
            raw_title = parts[1]
            m = re.search(r"\((\d{4})\)\s*$", raw_title)
            year = m.group(1) if m else "unknown"
            title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip()
            g_flags = parts[5:] if len(parts) > 5 else []
            genres_list = []
            for i, flag in enumerate(g_flags):
                if flag.strip() == "1" and i < len(genre_names):
                    genres_list.append(genre_names[i])
            genres_str = ", ".join(genres_list) if genres_list else "unknown"
            movies[movie_id] = {"title": title, "year": year, "genres": genres_str}
    return movies


def load_users(ml_dir: str) -> pd.DataFrame:
    """Загрузить демографию из u.user."""
    user_path = os.path.join(ml_dir, "u.user")
    df = pd.read_csv(
        user_path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    return df


def load_ratings(ml_dir: str) -> pd.DataFrame:
    """Загрузить рейтинги из u.data."""
    data_path = os.path.join(ml_dir, "u.data")
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df


def build_user_sequences(
    ml_dir: str,
    max_users: Optional[int] = None,
    max_events: int = 50,
    min_events: int = 5,
) -> Tuple[Dict[int, List[dict]], pd.DataFrame]:
    """Собрать последовательности событий по пользователям, отсортированные по времени."""
    logger = get_logger()

    movies = load_movies(ml_dir)
    ratings = load_ratings(ml_dir)
    users_df = load_users(ml_dir)

    ratings = ratings.sort_values(["user_id", "timestamp"])

    user_sequences: Dict[int, List[dict]] = defaultdict(list)
    for _, row in ratings.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        movie = movies.get(iid, {"title": "Unknown", "year": "unknown", "genres": "unknown"})
        event = {
            "title": movie["title"],
            "year": movie["year"],
            "genres": movie["genres"],
            "rating": int(row["rating"]),
            "timestamp": int(row["timestamp"]),
        }
        user_sequences[uid].append(event)

    filtered = {
        uid: events[:max_events]
        for uid, events in user_sequences.items()
        if len(events) >= min_events
    }

    if max_users is not None:
        uids = sorted(filtered.keys())[:max_users]
        filtered = {uid: filtered[uid] for uid in uids}

    valid_uids = set(filtered.keys())
    users_df = users_df[users_df["user_id"].isin(valid_uids)].copy()

    logger.info(
        f"Preprocessed: {len(filtered)} users, "
        f"avg {sum(len(v) for v in filtered.values()) / max(len(filtered), 1):.1f} events/user"
    )

    return filtered, users_df


if __name__ == "__main__":
    from src.data.movielens_download import download_movielens_100k

    ml_dir = download_movielens_100k()
    seqs, udf = build_user_sequences(ml_dir, max_users=10)
    for uid in list(seqs.keys())[:3]:
        print(f"User {uid}: {len(seqs[uid])} events, first: {seqs[uid][0]['title']}")
    print(udf.head())
