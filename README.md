# LLM4ES

Воспроизведение пайплайна из статьи **"LLM4ES: Learning User Embeddings from Event Sequences via Large Language Models"**.

## Пайплайн

1. **Сериализация** — последовательность событий пользователя в текст (заголовок + строки через `|`)
2. **Обогащение текста** — переписывание в разных форматах
3. **Fine-tuning** — дообучение LM на next-token prediction
4. **Извлечение эмбеддингов** — усреднение последних k=8 слоёв трансформера + mean pooling по токенам
5. **Downstream** — классификация на полученных эмбеддингах (LogReg)

Данные: MovieLens-100K, задача: предсказание пола пользователя.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
# базовый прогон на CPU
python -m scripts.run_pipeline --max_users 50

# с обогащением текста (rule-based)
python -m scripts.run_pipeline --serialization_mode mixed --max_users 50

# GPU + Qwen + LLM-обогащение + fine-tune
python -m scripts.run_pipeline --gpu --use-llm --serialization_mode mixed --finetune tiny_run --max_users 30
```

## Флаги

| Флаг | Что делает |
|---|---|
| `--gpu` | Использовать GPU, иначе переход на CPU |
| `--use-llm` | Обогащение текста через instruct LLM (Qwen) |
| `--llm_model` | default: `Qwen/Qwen2.5-0.5B-Instruct` |
| `--model_name` | default: `distilgpt2` на CPU, Qwen если `--use-llm` |
| `--serialization_mode` | `raw` — только сериализация; `mixed` — + обогащение |
| `--enrich_variants` | Сколько вариантов обогащения на пользователя (default: 3) |
| `--max_new_tokens` | Лимит генерации при LLM-обогащении (default: 256) |
| `--finetune` | `none` / `dry_run` (1-2 шага) / `tiny_run` (50 шагов) |
| `--save_ckpt` | Сохранить checkpoint после fine-tune |
| `--k_last_layers` | Сколько последних слоёв трансформера усреднять (default: 8) |
| `--max_length` | Макс. длина в токенах (default: 512) |
| `--batch_size` | Размер батча при извлечении эмбеддингов (default: 4) |
| `--max_users` | Сколько пользователей обрабатывать (default: 200) |
| `--task` | Downstream-задача: `gender` или `age_bucket` |
| `--seed` | Random seed (default: 123) |
