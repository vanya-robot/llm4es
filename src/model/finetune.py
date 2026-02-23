"""
Fine-tuning: next-token prediction (L_NTP).

Режимы:
  none     — пропустить
  dry_run  — 1-2 шага (проверка что код работает)
  tiny_run — 50-200 шагов на подмножестве
"""
import os
import time
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logging import get_logger, log_memory
from src.utils.config import FinetuneConfig


class TextDataset(Dataset):
    """Датасет токенизированных текстов для causal LM."""

    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def finetune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    config: FinetuneConfig,
    max_length: int = 512,
    device: str = "cpu",
) -> PreTrainedModel:
    """
    Дообучить causal LM на next-token prediction.
    labels = input_ids, сдвиг делает сама модель HF.
    """
    logger = get_logger()

    if config.mode == "none":
        logger.info("Fine-tuning mode = 'none', skipping.")
        return model

    if config.mode == "dry_run":
        num_steps = config.num_steps_dry
        logger.info(f"Fine-tuning mode = 'dry_run': {num_steps} steps")
    elif config.mode == "tiny_run":
        num_steps = config.num_steps_tiny
        texts = texts[:config.max_users_for_finetune]
        logger.info(f"Fine-tuning mode = 'tiny_run': {num_steps} steps on {len(texts)} texts")
    else:
        raise ValueError(f"Unknown finetune mode: {config.mode}")

    grad_accum = config.gradient_accumulation_steps if device == "cuda" else 1

    logger.info(
        f"Fine-tune config: device={device}, "
        f"grad_accum={grad_accum}, batch_size={config.batch_size}, "
        f"effective_batch={config.batch_size * grad_accum}"
    )

    # Модель в float32 для стабильного обучения на любом устройстве
    model = model.float()
    model.to(device)

    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    step = 0
    total_loss = 0.0
    accum_loss = 0.0
    micro_step = 0

    logger.info(
        f"Starting fine-tuning: {num_steps} optimizer steps, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}"
    )
    log_memory(logger)

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            t0 = time.time()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            accum_loss += loss.item() * grad_accum
            micro_step += 1

            if micro_step % grad_accum == 0 or step == num_steps - 1:
                optimizer.step()
                optimizer.zero_grad()

                step_time = time.time() - t0
                avg_step_loss = accum_loss / grad_accum
                total_loss += avg_step_loss
                step += 1

                logger.info(
                    f"  Step {step}/{num_steps} | "
                    f"loss={avg_step_loss:.4f} | "
                    f"time={step_time:.2f}s"
                )

                accum_loss = 0.0

                if step % 10 == 0:
                    log_memory(logger)

                if step >= num_steps:
                    break

    avg_loss = total_loss / max(step, 1)
    logger.info(f"Fine-tuning complete. Average loss: {avg_loss:.4f}")
    log_memory(logger)

    model.eval()

    if config.save_checkpoint:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(config.checkpoint_dir, f"finetuned_{timestamp}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")
    else:
        logger.info(
            f"Checkpoint saving disabled (--save_ckpt to enable). "
            f"Would save to: {config.checkpoint_dir}/"
        )

    return model
