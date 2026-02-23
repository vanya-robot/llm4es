"""
Извлечение эмбеддингов.

Формула:
    User_emb = MeanPooling( (1/k) * sum H_l )

hidden_states[0]  — выход embedding layer (исключаем)
hidden_states[1:] — выходы transformer-блоков

Берём последние k transformer-слоёв, усредняем по слоям,
затем mean pooling по токенам с учётом attention mask.
"""
from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.model.llm_wrapper import get_num_layers
from src.utils.logging import get_logger


def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling по токенам с маской. (B, T, H) -> (B, H)"""
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
    sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / sum_mask


def extract_embeddings_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    k_last_layers: int = 8,
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """
    Извлечь эмбеддинги для списка текстов.
    Возвращает numpy float32 массив (len(texts), hidden_dim).
    """
    logger = get_logger()
    model.eval()

    num_transformer_layers = get_num_layers(model)
    k = min(k_last_layers, num_transformer_layers)

    use_amp = (device == "cuda")

    logger.info(
        f"Extracting embeddings: {len(texts)} texts, "
        f"{num_transformer_layers} transformer layers, k={k}, "
        f"batch_size={batch_size}, device={device}"
    )

    all_embeddings = []
    _layer_check_done = False

    total_input_len = 0
    total_real_tokens = 0
    total_samples = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        total_input_len += input_ids.shape[1] * input_ids.shape[0]
        total_real_tokens += attention_mask.sum().item()
        total_samples += input_ids.shape[0]

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        hidden_states = outputs.hidden_states

        # [0] = embedding layer, [1:] = transformer blocks
        transformer_layers = hidden_states[1:]

        # Проверка размерностей (один раз)
        if not _layer_check_done:
            n_total = len(hidden_states)
            n_transformer = len(transformer_layers)
            layer_shape = transformer_layers[0].shape
            logger.info(
                f"  hidden_states: {n_total} "
                f"(1 emb + {n_transformer} transformer)"
            )
            logger.info(f"  layer shape: {list(layer_shape)}, k_used: {k}")
            if n_transformer != num_transformer_layers:
                logger.warning(
                    f"  Expected {num_transformer_layers} transformer layers, "
                    f"got {n_transformer}"
                )
            _layer_check_done = True

        last_k = transformer_layers[-k:]

        # Усреднение по слоям -> (batch, seq_len, hidden_dim)
        stacked = torch.stack(list(last_k), dim=0)
        layer_mean = stacked.float().mean(dim=0)

        # Mean pooling по токенам
        pooled = mean_pooling(layer_mean, attention_mask)

        all_embeddings.append(pooled.cpu().numpy())

        if (i // batch_size) % 10 == 0 and i > 0:
            logger.info(f"  Processed {i + len(batch_texts)}/{len(texts)} texts")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    avg_input_len = total_input_len / max(total_samples, 1)
    avg_real_tokens = total_real_tokens / max(total_samples, 1)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(
        f"Token stats: avg input_ids={avg_input_len:.1f}, "
        f"avg real tokens={avg_real_tokens:.1f}"
    )

    assert not np.isnan(embeddings).any(), "NaN in embeddings!"
    assert embeddings.shape[0] == len(texts), "Embedding count mismatch!"

    return embeddings


def extract_single_embedding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    k_last_layers: int = 8,
    max_length: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Эмбеддинг для одного текста. Возвращает 1D numpy."""
    emb = extract_embeddings_batch(
        model, tokenizer, [text],
        k_last_layers=k_last_layers,
        max_length=max_length,
        batch_size=1,
        device=device,
    )
    return emb[0]
