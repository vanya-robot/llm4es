#!/usr/bin/env python3
"""
Экспорт артефактов: эмбеддинги (.npy) и метрики (.json).
"""
import sys
import os
import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import get_logger, timer
from src.model.llm_wrapper import resolve_device
from src.data.movielens_download import download_movielens_100k
from src.data.preprocess_movielens import build_user_sequences
from src.serialization.serialize import serialize_events
from src.model.llm_wrapper import load_model_and_tokenizer
from src.model.embed import extract_embeddings_batch
from src.downstream.classify import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Export LLM4ES artifacts")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--k_last_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LLM4ES: Export Artifacts")
    logger.info("=" * 60)

    ml_dir = download_movielens_100k()
    user_sequences, users_df = build_user_sequences(ml_dir, max_users=args.max_users)
    user_ids = sorted(user_sequences.keys())

    texts = [serialize_events(user_sequences[uid]) for uid in user_ids]

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    with timer("Embedding extraction", logger):
        embeddings = extract_embeddings_batch(
            model, tokenizer, texts,
            k_last_layers=args.k_last_layers,
            batch_size=4,
            device=device,
        )

    emb_path = os.path.join(args.output_dir, "user_embeddings.npy")
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings: {emb_path} -- shape {embeddings.shape}")

    ids_path = os.path.join(args.output_dir, "user_ids.json")
    with open(ids_path, "w") as f:
        json.dump(user_ids, f)
    logger.info(f"Saved user IDs: {ids_path}")

    metrics = train_and_evaluate(embeddings, user_ids, users_df, task="gender")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics: {metrics_path}")

    logger.info("")
    logger.info("Exported:")
    logger.info(f"  {emb_path} -- {embeddings.shape}")
    logger.info(f"  {ids_path} -- {len(user_ids)} users")
    logger.info(f"  {metrics_path} -- accuracy={metrics.get('accuracy', 'N/A'):.4f}")

    try:
        import pandas as pd
        df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
        df.insert(0, "user_id", user_ids)
        pq_path = os.path.join(args.output_dir, "user_embeddings.parquet")
        df.to_parquet(pq_path, index=False)
        logger.info(f"  {pq_path} -- parquet")
    except Exception as e:
        logger.info(f"  Parquet skipped: {e}")


if __name__ == "__main__":
    main()
