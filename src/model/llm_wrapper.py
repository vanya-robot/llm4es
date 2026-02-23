"""
Загрузка causal LM и токенизатора из HuggingFace.
На GPU грузит в float16, на CPU — float32.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from src.utils.logging import get_logger


def resolve_device(gpu_requested: bool) -> str:
    """Определить устройство. Если CUDA недоступна — fallback на CPU."""
    logger = get_logger()
    if gpu_requested:
        if torch.cuda.is_available():
            logger.info(f"[DEVICE] cuda ({torch.cuda.get_device_name(0)})")
            return "cuda"
        else:
            logger.warning(
                "[DEVICE] --gpu requested but CUDA is not available. "
                "Falling back to CPU."
            )
            return "cpu"
    logger.info("[DEVICE] cpu")
    return "cpu"


def load_model_and_tokenizer(
    model_name: str = "distilgpt2",
    device: str = "cpu",
) -> tuple:
    """Загрузить causal LM и токенизатор."""
    logger = get_logger()
    logger.info(f"Loading model '{model_name}' on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,   # нужно для извлечения эмбеддингов
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else "?"
    logger.info(
        f"Model loaded: {n_params / 1e6:.1f}M params, "
        f"{n_layers} layers, hidden_size={model.config.hidden_size}, "
        f"dtype={dtype}"
    )

    return model, tokenizer


def get_num_layers(model: PreTrainedModel) -> int:
    """Число transformer-слоёв модели."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    else:
        raise ValueError("Cannot determine number of layers from model config.")
