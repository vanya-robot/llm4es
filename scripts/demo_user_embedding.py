#!/usr/bin/env python3
"""
Демо: эмбеддинг одного пользователя.
Выводит текст, enriched-варианты, эмбеддинг, top-5 похожих.
"""
import sys
import os
import argparse

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import get_logger, timer
from src.model.llm_wrapper import resolve_device
from src.data.movielens_download import download_movielens_100k
from src.data.preprocess_movielens import build_user_sequences
from src.serialization.serialize import serialize_events, serialize_user
from src.enrichment.enrich import enrich
from src.model.llm_wrapper import load_model_and_tokenizer
from src.model.embed import extract_embeddings_batch, extract_single_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="LLM4ES Demo: User Embedding")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--user_id", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--k_last_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.gpu)

    logger.info("=" * 60)
    logger.info("LLM4ES Demo: User Embedding")
    logger.info("=" * 60)

    ml_dir = download_movielens_100k()
    user_sequences, users_df = build_user_sequences(
        ml_dir, max_users=args.max_users, max_events=30, min_events=5
    )
    user_ids = sorted(user_sequences.keys())

    demo_uid = args.user_id if args.user_id and args.user_id in user_sequences else user_ids[0]
    events = user_sequences[demo_uid]

    print("\n" + "=" * 60)
    print(f"USER {demo_uid} -- Serialized ({len(events)} events)")
    print("=" * 60)
    serialized = serialize_user(demo_uid, events)
    print(serialized[:800])
    if len(serialized) > 800:
        print(f"... ({len(serialized)} chars total)")

    print("\n" + "=" * 60)
    print("ENRICHED VARIANTS")
    print("=" * 60)

    for mode in ["markdown_table", "bullet_list", "summary"]:
        print(f"\n--- {mode} ---")
        enriched = enrich(events, mode=mode)
        print(enriched[:500])
        if len(enriched) > 500:
            print(f"... ({len(enriched)} chars)")

    print("\n" + "=" * 60)
    print("EMBEDDING")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    emb = extract_single_embedding(
        model, tokenizer, serialized,
        k_last_layers=args.k_last_layers,
        device=device,
    )

    print(f"\nShape: {emb.shape}")
    print(f"Dtype: {emb.dtype}")
    print(f"Norm:  {np.linalg.norm(emb):.4f}")
    print(f"First 5: {emb[:5]}")
    print(f"Min/Max: {emb.min():.4f} / {emb.max():.4f}")

    print("\n" + "=" * 60)
    print("TOP-5 SIMILAR USERS")
    print("=" * 60)

    all_texts = [serialize_events(user_sequences[uid]) for uid in user_ids]

    with timer("Batch embedding extraction", logger):
        all_embs = extract_embeddings_batch(
            model, tokenizer, all_texts,
            k_last_layers=args.k_last_layers,
            batch_size=4,
            device=device,
        )

    demo_idx = user_ids.index(demo_uid)
    demo_emb = all_embs[demo_idx].reshape(1, -1)

    sims = cosine_similarity(demo_emb, all_embs)[0]
    sims[demo_idx] = -1.0
    top5_indices = np.argsort(sims)[::-1][:5]

    for rank, idx in enumerate(top5_indices, 1):
        uid = user_ids[idx]
        print(f"  {rank}. User {uid} (sim={sims[idx]:.4f}, {len(user_sequences[uid])} events)")

    # Диагностика cosine similarity
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY DIAGNOSTICS")
    print("=" * 60)
    all_sims = cosine_similarity(all_embs)
    np.fill_diagonal(all_sims, 0.0)
    n = len(user_ids)
    mean_sim = all_sims.sum() / (n * (n - 1))
    max_sim = all_sims.max()
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Max:  {max_sim:.4f}")
    if mean_sim > 0.95:
        print("  WARNING: embeddings very similar, check max_length/truncation.")

    # Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    print(f"  NaN: {np.isnan(emb).any()}")
    print(f"  Finite: {np.isfinite(emb).all()}")

    emb2 = extract_single_embedding(
        model, tokenizer, serialized,
        k_last_layers=args.k_last_layers,
        device=device,
    )
    diff = np.max(np.abs(emb - emb2))
    print(f"  Deterministic (max diff): {diff:.2e}")
    print(f"  OK" if diff < 1e-5 and not np.isnan(emb).any() else "  FAIL")


if __name__ == "__main__":
    main()
