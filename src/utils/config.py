"""
Конфигурация пайплайна (dataclass).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset_name: str = "movielens-100k"
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    max_users: Optional[int] = None
    max_events_per_user: int = 50
    min_events_per_user: int = 5
    seed: int = 42


@dataclass
class SerializationConfig:
    separator: str = " | "
    event_separator: str = "\n"
    include_header: bool = True


@dataclass
class EnrichmentConfig:
    """Обогащение текста (data augmentation).
    На CPU — rule-based, на GPU — через Qwen instruct.
    """
    modes: list = field(default_factory=lambda: ["markdown_table", "bullet_list", "summary", "json_like"])
    num_variants: int = 3
    use_llm: bool = False
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 256


@dataclass
class ModelConfig:
    model_name: str = "distilgpt2"
    max_length: int = 512
    device: str = "cpu"
    k_last_layers: int = 8          # сколько последних слоёв усреднять
    output_hidden_states: bool = True
    batch_size: int = 4


@dataclass
class FinetuneConfig:
    mode: str = "none"              # "none" | "dry_run" | "tiny_run"
    learning_rate: float = 5e-5
    num_steps_dry: int = 2
    num_steps_tiny: int = 50
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_users_for_finetune: int = 50
    save_checkpoint: bool = False
    checkpoint_dir: str = "artifacts/checkpoints"


@dataclass
class DownstreamConfig:
    task: str = "gender"            # "gender" | "age_bucket"
    test_size: float = 0.2
    seed: int = 42


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    seed: int = 42
    verbose: bool = True
