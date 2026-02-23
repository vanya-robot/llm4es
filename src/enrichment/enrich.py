"""
Единый интерфейс для rule-based обогащения текста.
enrich() — один формат, enrich_multi() — несколько вариантов.
"""
from typing import List, Dict, Optional
import random

from src.enrichment.rules_enrichment import ENRICHMENT_FUNCTIONS
from src.serialization.serialize import serialize_events


def enrich(events: List[Dict], mode: str = "bullet_list") -> str:
    """Переформатировать события в указанный текстовый формат."""
    if mode == "original":
        return serialize_events(events)

    func = ENRICHMENT_FUNCTIONS.get(mode)
    if func is None:
        raise ValueError(
            f"Unknown enrichment mode '{mode}'. "
            f"Available: {list(ENRICHMENT_FUNCTIONS.keys()) + ['original']}"
        )
    return func(events)


def enrich_multi(
    events: List[Dict],
    modes: Optional[List[str]] = None,
    num_variants: int = 3,
    seed: Optional[int] = None,
) -> List[str]:
    """Сгенерировать несколько вариантов текста из одной последовательности событий."""
    if modes is None:
        modes = ["original"] + list(ENRICHMENT_FUNCTIONS.keys())

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    selected = [rng.choice(modes) for _ in range(num_variants)]

    variants = []
    for mode in selected:
        variants.append(enrich(events, mode))

    return variants
