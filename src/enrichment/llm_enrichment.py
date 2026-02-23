"""
LLM-based обогащение текста через instruct-модель (Qwen2.5-0.5B-Instruct).
Переписывает сериализованные события в разные форматы через генерацию.
Результаты кэшируются на диск.
"""
import os
import json
import hashlib
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.utils.logging import get_logger


# Промпты для разных стилей переписывания
VARIANT_PROMPTS = {
    "markdown_table": (
        "Rewrite the following user viewing history as a Markdown table "
        "with columns: Title, Year, Genres, Rating, Date. "
        "Keep all information, do not add anything extra."
    ),
    "bullet_list": (
        "Rewrite the following user viewing history as a bullet list, "
        "grouped by genre or rating. Preserve all information."
    ),
    "narrative": (
        "Summarize the following user viewing history as a short narrative paragraph "
        "describing the user's movie taste, preferences, and behavior. "
        "Keep it factual and based on the data."
    ),
    "json_like": (
        "Rewrite the following user viewing history in a JSON-like structured format. "
        "Use keys: title, year, genres, rating, date for each entry."
    ),
}

DEFAULT_VARIANT_ORDER = ["markdown_table", "bullet_list", "narrative", "json_like"]


def load_instruct_llm(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Загрузить instruct-модель для обогащения."""
    logger = get_logger()
    logger.info(f"Loading instruct LLM '{model_id}' on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instruct LLM loaded: {n_params / 1e6:.1f}M params, dtype={dtype}")
    return tokenizer, model


def enrich_with_llm(
    serialized_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    variant_prompt: str,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> str:
    """Переписать текст с помощью instruct-модели по заданному промпту."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that reformats data."},
        {"role": "user", "content": f"{variant_prompt}\n\n{serialized_text}"},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = (
            f"System: {messages[0]['content']}\n"
            f"User: {messages[1]['content']}\n"
            f"Assistant:"
        )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def enrich_multi_llm(
    serialized_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    n_variants: int = 3,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> List[str]:
    """Сгенерировать n_variants переписанных версий текста в разных стилях."""
    styles = DEFAULT_VARIANT_ORDER
    selected_styles = [styles[i % len(styles)] for i in range(n_variants)]

    variants = []
    for style in selected_styles:
        prompt = VARIANT_PROMPTS[style]
        text = enrich_with_llm(
            serialized_text, tokenizer, model, prompt,
            max_new_tokens=max_new_tokens, device=device,
        )
        variants.append(text)
    return variants


def combine_variants(variants: List[str]) -> str:
    """Объединить варианты с заголовками секций."""
    parts = []
    style_names = DEFAULT_VARIANT_ORDER
    for i, text in enumerate(variants):
        style = style_names[i % len(style_names)].replace("_", " ").title()
        parts.append(f"### Variant {i + 1} ({style})\n{text}")
    return "\n\n".join(parts)


# --- Кэш на диск ---
_CACHE_PATH = "artifacts/enriched_cache.jsonl"


def _cache_key(user_id: int, serialized_text: str, n_variants: int, model_id: str) -> str:
    raw = f"{user_id}|{n_variants}|{model_id}|{serialized_text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache() -> dict:
    cache = {}
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["key"]] = entry["variants"]
    return cache


def save_cache_entry(key: str, variants: List[str]):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "a") as f:
        f.write(json.dumps({"key": key, "variants": variants}) + "\n")


def enrich_users_with_llm(
    user_ids: List[int],
    serialized_texts: dict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    n_variants: int = 3,
    max_new_tokens: int = 256,
    device: str = "cpu",
    model_id: str = "unknown",
) -> dict:
    """Обогатить всех пользователей через LLM. Использует дисковый кэш."""
    logger = get_logger()
    cache = load_cache()
    results = {}
    cache_hits = 0

    for uid in tqdm(user_ids, desc="LLM enrichment"):
        key = _cache_key(uid, serialized_texts[uid], n_variants, model_id)
        if key in cache:
            results[uid] = cache[key]
            cache_hits += 1
        else:
            variants = enrich_multi_llm(
                serialized_texts[uid], tokenizer, model,
                n_variants=n_variants,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            results[uid] = variants
            save_cache_entry(key, variants)

    logger.info(
        f"LLM enrichment done: {len(user_ids)} users, "
        f"{cache_hits} cache hits, {len(user_ids) - cache_hits} generated"
    )
    return results
