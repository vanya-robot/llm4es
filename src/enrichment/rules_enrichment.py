"""
Rule-based обогащение текста: переписывание событий в разных форматах
(markdown, bullets, summary, json).
"""
import json
from typing import List, Dict

from src.serialization.mappers import rating_to_text, map_genres
from src.serialization.serialize import format_timestamp


def to_markdown_table(events: List[Dict]) -> str:
    header = "| Title | Year | Genres | Rating | Date |"
    sep = "|-------|------|--------|--------|------|"
    rows = []
    for ev in events:
        row = "| {} | {} | {} | {} | {} |".format(
            ev.get("title", "Unknown"),
            ev.get("year", "?"),
            map_genres(ev.get("genres", "")),
            rating_to_text(ev.get("rating", 3)),
            format_timestamp(ev.get("timestamp", 0)),
        )
        rows.append(row)
    return "\n".join([header, sep] + rows)


def to_bullet_list(events: List[Dict]) -> str:
    lines = ["Movie viewing history:"]
    for i, ev in enumerate(events, 1):
        line = (
            f"  {i}. \"{ev.get('title', 'Unknown')}\" ({ev.get('year', '?')}) "
            f"-- rated {rating_to_text(ev.get('rating', 3))} "
            f"on {format_timestamp(ev.get('timestamp', 0))}. "
            f"Genres: {map_genres(ev.get('genres', ''))}."
        )
        lines.append(line)
    return "\n".join(lines)


def to_summary(events: List[Dict]) -> str:
    if not events:
        return "No viewing history available."
    n = len(events)
    ratings = [ev.get("rating", 3) for ev in events]
    avg_rating = sum(ratings) / len(ratings)
    all_genres = set()
    for ev in events:
        for g in ev.get("genres", "").split(","):
            g = g.strip()
            if g and g != "unknown":
                all_genres.add(g)
    top_genres = ", ".join(sorted(all_genres)[:5]) if all_genres else "various"
    first_title = events[0].get("title", "Unknown")
    last_title = events[-1].get("title", "Unknown")

    summary = (
        f"This user watched {n} movies. "
        f"Their average rating is {avg_rating:.1f}/5. "
        f"Preferred genres include {top_genres}. "
        f"Their history starts with \"{first_title}\" and most recently includes \"{last_title}\"."
    )
    return summary


def to_json_like(events: List[Dict]) -> str:
    records = []
    for ev in events:
        records.append({
            "title": ev.get("title", "Unknown"),
            "year": ev.get("year", "?"),
            "genres": map_genres(ev.get("genres", "")),
            "rating": rating_to_text(ev.get("rating", 3)),
            "date": format_timestamp(ev.get("timestamp", 0)),
        })
    return json.dumps(records, indent=2, ensure_ascii=False)


ENRICHMENT_FUNCTIONS = {
    "markdown_table": to_markdown_table,
    "bullet_list": to_bullet_list,
    "summary": to_summary,
    "json_like": to_json_like,
}
