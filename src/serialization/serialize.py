"""
Сериализация: список событий -> текст.

Пример:
  Title | Year | Genres | Rating | Date
  Toy Story | 1995 | Animation, Children's, Comedy | excellent | 1998-01-05
"""
from datetime import datetime
from typing import List, Dict

from src.serialization.mappers import rating_to_text, map_genres


def format_timestamp(ts: int) -> str:
    """Unix timestamp -> строка даты."""
    try:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except (OSError, ValueError):
        return "unknown-date"


def serialize_events(
    events: List[Dict],
    separator: str = " | ",
    event_separator: str = "\n",
    include_header: bool = True,
) -> str:
    """Сериализовать последовательность событий в текстовый блок."""
    lines = []

    if include_header:
        header = separator.join(["Title", "Year", "Genres", "Rating", "Date"])
        lines.append(header)

    for ev in events:
        fields = [
            ev.get("title", "Unknown"),
            str(ev.get("year", "unknown")),
            map_genres(ev.get("genres", "unknown")),
            rating_to_text(ev.get("rating", 3)),
            format_timestamp(ev.get("timestamp", 0)),
        ]
        lines.append(separator.join(fields))

    return event_separator.join(lines)


def serialize_user(
    user_id: int,
    events: List[Dict],
    separator: str = " | ",
    event_separator: str = "\n",
    include_header: bool = True,
) -> str:
    """Сериализация с заголовком пользователя."""
    body = serialize_events(events, separator, event_separator, include_header)
    return f"User {user_id} viewing history:\n{body}"
