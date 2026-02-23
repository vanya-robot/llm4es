#!/usr/bin/env python3
"""
Полный пайплайн LLM4ES: данные -> сериализация -> (обогащение) -> (fine-tune) -> эмбеддинги -> классификация.
"""
import sys
import os
import argparse
import time

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import PipelineConfig
from src.utils.logging import get_logger, timer, log_memory
from src.model.llm_wrapper import resolve_device
from src.data.movielens_download import download_movielens_100k
from src.data.preprocess_movielens import build_user_sequences
from src.serialization.serialize import serialize_events
from src.enrichment.enrich import enrich_multi
from src.model.llm_wrapper import load_model_and_tokenizer
from src.model.finetune import finetune
from src.model.embed import extract_embeddings_batch
from src.downstream.classify import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="LLM4ES Pipeline")

    parser.add_argument("--gpu", action="store_true",
                        help="Использовать GPU если доступен")
    parser.add_argument("--serialization_mode", type=str, default="raw",
                        choices=["raw", "mixed"])
    parser.add_argument("--use-llm", action="store_true",
                        help="Обогащение через instruct LLM (Qwen)")
    parser.add_argument("--llm_model", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--enrich_variants", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Модель для fine-tune + эмбеддингов")
    parser.add_argument("--finetune", type=str, default="none",
                        choices=["none", "dry_run", "tiny_run"])
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--k_last_layers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_users", type=int, default=200)
    parser.add_argument("--task", type=str, default="gender",
                        choices=["gender", "age_bucket"])
    parser.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def compute_text_stats(texts, tokenizer, max_length):
    total_chars = sum(len(t) for t in texts)
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=max_length)
        total_tokens += len(enc["input_ids"])
    n = max(len(texts), 1)
    return {
        "num_texts": len(texts),
        "avg_chars": total_chars / n,
        "avg_tokens": total_tokens / n,
    }


def main():
    args = parse_args()
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("LLM4ES Pipeline")
    logger.info("=" * 60)

    device = resolve_device(args.gpu)

    # Выбор модели
    if args.model_name is not None:
        model_name = args.model_name
    elif args.use_llm:
        model_name = args.llm_model
    else:
        model_name = "distilgpt2"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = PipelineConfig()
    cfg.data.max_users = args.max_users
    cfg.model.model_name = model_name
    cfg.model.device = device
    cfg.model.k_last_layers = args.k_last_layers
    cfg.model.max_length = args.max_length
    cfg.model.batch_size = args.batch_size
    cfg.finetune.mode = args.finetune
    cfg.finetune.save_checkpoint = args.save_ckpt
    cfg.downstream.task = args.task
    cfg.enrichment.num_variants = args.enrich_variants
    cfg.enrichment.use_llm = args.use_llm
    cfg.enrichment.llm_model = args.llm_model
    cfg.enrichment.max_new_tokens = args.max_new_tokens
    cfg.seed = args.seed

    ser_mode = args.serialization_mode

    logger.info(f"[DEVICE] {device}")
    logger.info(f"[MODE] serialization_mode={ser_mode}")
    if ser_mode == "mixed":
        if args.use_llm:
            logger.info(
                f"[ENRICH] LLM (model={args.llm_model}, "
                f"variants={cfg.enrichment.num_variants}, "
                f"max_new_tokens={cfg.enrichment.max_new_tokens})"
            )
        else:
            logger.info(
                f"[ENRICH] RULES (variants={cfg.enrichment.num_variants})"
            )
    else:
        logger.info("[ENRICH] none (raw mode)")
    logger.info(f"[MODEL] {model_name}")

    timings = {}
    pipeline_start = time.time()

    # 1. Скачивание данных
    with timer("Stage 1: Download", logger):
        t0 = time.time()
        ml_dir = download_movielens_100k(cfg.data.data_dir)
        timings["download"] = time.time() - t0

    # 2. Предобработка
    with timer("Stage 2: Preprocess", logger):
        t0 = time.time()
        user_sequences, users_df = build_user_sequences(
            ml_dir,
            max_users=cfg.data.max_users,
            max_events=cfg.data.max_events_per_user,
            min_events=cfg.data.min_events_per_user,
        )
        user_ids = sorted(user_sequences.keys())
        timings["preprocess"] = time.time() - t0

    logger.info(f"Users: {len(user_ids)}, events example: {len(user_sequences[user_ids[0]])}")

    # 3. Сериализация
    with timer("Stage 3: Serialize", logger):
        t0 = time.time()
        serialized_texts = {}
        for uid in user_ids:
            serialized_texts[uid] = serialize_events(
                user_sequences[uid],
                separator=cfg.serialization.separator,
                include_header=cfg.serialization.include_header,
            )
        timings["serialize"] = time.time() - t0

    logger.info(f"Text example (user {user_ids[0]}, first 200 chars):")
    logger.info(serialized_texts[user_ids[0]][:200] + "...")

    # 4. Формирование текстов (raw / mixed)
    with timer("Stage 4: Build texts", logger):
        t0 = time.time()
        all_texts = []
        text_to_user = []

        if ser_mode == "raw":
            for uid in user_ids:
                all_texts.append(serialized_texts[uid])
                text_to_user.append(uid)
        else:
            if args.use_llm:
                from src.enrichment.llm_enrichment import (
                    load_instruct_llm,
                    enrich_users_with_llm,
                )

                enrich_tokenizer, enrich_model = load_instruct_llm(
                    model_id=args.llm_model,
                    device=device,
                )

                enriched_map = enrich_users_with_llm(
                    user_ids=user_ids,
                    serialized_texts=serialized_texts,
                    tokenizer=enrich_tokenizer,
                    model=enrich_model,
                    n_variants=cfg.enrichment.num_variants,
                    max_new_tokens=cfg.enrichment.max_new_tokens,
                    device=device,
                    model_id=args.llm_model,
                )

                if args.llm_model != model_name:
                    del enrich_model, enrich_tokenizer
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    logger.info("Freed enrichment model memory")

                for uid in user_ids:
                    all_enriched = [serialized_texts[uid]] + enriched_map[uid]
                    for text in all_enriched:
                        all_texts.append(text)
                        text_to_user.append(uid)
            else:
                for uid in user_ids:
                    events = user_sequences[uid]
                    variants = enrich_multi(
                        events,
                        num_variants=cfg.enrichment.num_variants,
                        seed=cfg.seed + uid,
                    )
                    all_enriched = [serialized_texts[uid]] + variants
                    for text in all_enriched:
                        all_texts.append(text)
                        text_to_user.append(uid)

        timings["build_texts"] = time.time() - t0

    texts_per_user = len(all_texts) / max(len(user_ids), 1)
    logger.info(
        f"Total texts: {len(all_texts)} "
        f"({len(user_ids)} users, ~{texts_per_user:.1f} texts/user)"
    )

    # 5. Загрузка модели
    with timer("Stage 5: Load model", logger):
        t0 = time.time()
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        timings["load_model"] = time.time() - t0

    log_memory(logger)

    text_stats = compute_text_stats(all_texts, tokenizer, cfg.model.max_length)
    logger.info(
        f"Text stats: {text_stats['num_texts']} texts, "
        f"avg {text_stats['avg_chars']:.0f} chars, "
        f"avg {text_stats['avg_tokens']:.0f} tokens"
    )

    # 5b. Fine-tune
    with timer(f"Stage 5b: Fine-tune (mode={cfg.finetune.mode})", logger):
        t0 = time.time()
        model = finetune(
            model, tokenizer, all_texts, cfg.finetune,
            max_length=cfg.model.max_length,
            device=device,
        )
        timings["finetune"] = time.time() - t0

    # 6. Эмбеддинги
    with timer("Stage 6: Extract embeddings", logger):
        t0 = time.time()
        all_embeddings = extract_embeddings_batch(
            model, tokenizer, all_texts,
            k_last_layers=cfg.model.k_last_layers,
            max_length=cfg.model.max_length,
            batch_size=cfg.model.batch_size,
            device=device,
        )
        timings["embed"] = time.time() - t0

    logger.info(f"All embeddings shape: {all_embeddings.shape}")

    # 6b. Усреднение эмбеддингов по пользователю
    with timer("Stage 6b: Average per user", logger):
        t0 = time.time()
        user_embeddings = {}
        for i, uid in enumerate(text_to_user):
            if uid not in user_embeddings:
                user_embeddings[uid] = []
            user_embeddings[uid].append(all_embeddings[i])

        emb_matrix = np.zeros((len(user_ids), all_embeddings.shape[1]), dtype=np.float32)
        for j, uid in enumerate(user_ids):
            emb_matrix[j] = np.mean(user_embeddings[uid], axis=0)
        timings["avg_embeddings"] = time.time() - t0

    logger.info(f"User embedding matrix: {emb_matrix.shape}")

    # Cosine similarity
    if len(user_ids) > 1:
        sims = cosine_similarity(emb_matrix)
        np.fill_diagonal(sims, 0.0)
        mean_sim = sims.sum() / (len(user_ids) * (len(user_ids) - 1))
        max_sim = sims.max()
        logger.info(f"Cosine similarity: mean={mean_sim:.4f}, max={max_sim:.4f}")
        if mean_sim > 0.95:
            logger.warning(
                "Embeddings highly similar (mean cosine > 0.95). "
                "Check truncation or max_length."
            )

    # 7. Downstream
    with timer("Stage 7: Downstream classification", logger):
        t0 = time.time()
        metrics = train_and_evaluate(
            emb_matrix, user_ids, users_df,
            task=cfg.downstream.task,
            test_size=cfg.downstream.test_size,
            seed=cfg.downstream.seed,
        )
        timings["downstream"] = time.time() - t0

    # Итоговый отчёт
    total_time = time.time() - pipeline_start
    timings["total"] = total_time

    num_transformer_layers = model.config.num_hidden_layers
    k_used = min(cfg.model.k_last_layers, num_transformer_layers)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE REPORT")
    logger.info("=" * 60)
    logger.info(f"Device:             {device}")
    logger.info(f"Model:              {model_name}")
    logger.info(f"Users:              {len(user_ids)}")
    logger.info(f"Serialization mode: {ser_mode}")
    if ser_mode == "mixed":
        logger.info(f"Enrichment:         {'LLM (' + args.llm_model + ')' if args.use_llm else 'RULES'}")
    logger.info(f"Texts per user:     {texts_per_user:.1f}")
    logger.info(f"Avg text length:    {text_stats['avg_chars']:.0f} chars / {text_stats['avg_tokens']:.0f} tokens")
    logger.info(f"Embedding dim:      {emb_matrix.shape[1]}")
    logger.info(f"Transformer layers: {num_transformer_layers}")
    logger.info(f"k (last layers):    {k_used}")
    logger.info(f"Fine-tune mode:     {cfg.finetune.mode}")
    logger.info(f"Task:               {cfg.downstream.task}")
    logger.info(f"Accuracy:           {metrics.get('accuracy', 'N/A')}")
    roc_auc = metrics.get("roc_auc")
    logger.info(f"ROC-AUC:            {roc_auc if roc_auc is not None else 'N/A'}")
    logger.info("")
    logger.info("Timings:")
    for stage, t in timings.items():
        logger.info(f"  {stage:20s}: {t:.2f}s")
    logger.info("=" * 60)

    log_memory(logger)

    return metrics, emb_matrix, timings


if __name__ == "__main__":
    main()
