import logging
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
from huggingface_hub import login

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG FILES
# =============================================================================

DEFAULT_CONFIG_FILE = ".train_default.json"

TASK_CONFIG_FILES = {
    "mlm": ".train_mlm.json",
    "classification": ".train_classification.json",
    "ner": ".train_ner.json",
}

DEFAULT_CONFIG = {
    # Model
    "model_name": "distilbert/distilbert-base-uncased",
    "output_dir": "output",
    # Data
    "dataset_path": "training_data.parquet",
    "eval_split_ratio": 0.05,
    "preprocessing_workers": 8,
    "keep_columns": [],
    # Training
    "num_train_epochs": 3,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": True,
    "dataloader_num_workers": 4,
    # Checkpointing
    "eval_steps": 500,
    "save_steps": 1000,
    "save_total_limit": 3,
    "resume_from_checkpoint": None,
    # Hub
    "push_to_hub": False,
    "hub_model_id": None,
}

TASK_DEFAULTS = {
    "mlm": {
        "output_dir": "distilbert-dapt",
        "mlm_probability": 0.15,
        "block_size": 512,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        
    },
    "classification": {
        "output_dir": "distilbert-classification",
        "classification_column": "label",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
    },
    "ner": {
        "output_dir": "distilbert-ner",
        "ner_column": "ner_tags",
        "words_column": "words",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        # Label set — override in task config if needed
        "labels": [
            "O",
        ],
    },
}


# =============================================================================
# CONFIG LOADING
# =============================================================================


def _write_default(path: Path, content: dict) -> None:
    with open(path, "w") as f:
        json.dump(content, f, indent=4)
    logger.info(f"Created default config: {path}. Edit and rerun.")


def load_configs(task: str) -> Optional[SimpleNamespace]:
    """
    Loads and merges configs in priority order (lowest → highest):
      1. Hardcoded DEFAULT_CONFIG
      2. Hardcoded TASK_DEFAULTS[task]
      3. .train_default.json  (user global overrides)
      4. .train_<task>.json   (user task-specific overrides)

    Creates missing files with sensible defaults on first run.
    Returns None if any config file was newly created (prompt user to edit).
    """
    if task not in TASK_CONFIG_FILES:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {list(TASK_CONFIG_FILES)}"
        )

    default_path = Path(DEFAULT_CONFIG_FILE)
    task_path = Path(TASK_CONFIG_FILES[task])

    created_any = False

    # Create global default if missing
    if not default_path.exists():
        _write_default(default_path, DEFAULT_CONFIG)
        created_any = True

    # Create task default if missing
    if not task_path.exists():
        task_content = {**DEFAULT_CONFIG, **TASK_DEFAULTS[task]}
        _write_default(task_path, task_content)
        created_any = True

    if created_any:
        return None

    # Load and merge
    with open(default_path) as f:
        global_cfg = json.load(f)

    with open(task_path) as f:
        task_cfg = json.load(f)

    merged = {**DEFAULT_CONFIG, **TASK_DEFAULTS[task], **global_cfg, **task_cfg}
    merged["task"] = task

    logger.info(f"Config loaded — task={task}")
    logger.info(f"  model_name   : {merged['model_name']}")
    logger.info(f"  output_dir   : {merged['output_dir']}")
    logger.info(f"  dataset_path : {merged['dataset_path']}")

    return SimpleNamespace(**merged)


# =============================================================================
# HF LOGIN
# =============================================================================


def maybe_login() -> None:
    if Path("hf_token").exists():
        with open("hf_token") as f:
            token = f.read().strip()
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.info("No hf_token found — proceeding without Hub capability")


# =============================================================================
# SHARED HELPERS
# =============================================================================


def load_and_split(args: SimpleNamespace):
    """Load dataset from parquet and optionally split into train/eval."""
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset):,} examples from {args.dataset_path}")

    keep_cols = getattr(args, "keep_columns", None)
    if keep_cols:
        keep_set = set(keep_cols)
        drop_cols = [c for c in dataset.column_names if c not in keep_set]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
            logger.info(f"Dropped columns: {drop_cols}")

    eval_dataset = None
    ratio = getattr(args, "eval_split_ratio", 0.05)
    if ratio and ratio > 0:
        split = dataset.train_test_split(test_size=ratio, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Split → {len(dataset):,} train | {len(eval_dataset):,} eval")

    return dataset, eval_dataset


def build_training_args(args: SimpleNamespace, has_eval: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=False,
        fp16=getattr(args, "fp16", True),
        dataloader_num_workers=getattr(args, "dataloader_num_workers", 4),
        report_to="tensorboard",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or args.output_dir,
        hub_strategy="end",
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def save_outputs(
    trainer: Trainer, tokenizer, args: SimpleNamespace, extra: Optional[dict] = None
) -> None:
    logger.info(f"Saving model → {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if extra:
        for filename, content in extra.items():
            out_path = Path(args.output_dir) / filename
            with open(out_path, "w") as f:
                json.dump(content, f, indent=4)
            logger.info(f"Saved {filename}")

    if args.push_to_hub:
        logger.info(f"Pushing to Hub: {args.hub_model_id or args.output_dir}")
        trainer.push_to_hub()


# =============================================================================
# TASK: MLM / DAPT
# =============================================================================


def run_mlm(args: SimpleNamespace) -> None:
    """
    Masked Language Modeling for Domain-Adaptive Pre-Training.
    Dataset: parquet with a 'text' column.
    """
    logger.info("=== Task: MLM (DAPT) ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, eval_dataset = load_and_split(args)

    block_size = getattr(args, "block_size", 512)

    def tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            return_special_tokens_mask=True,
        )

        return tokenized

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated[list(examples.keys())[0]]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tok_kwargs = dict(
        batched=True, num_proc=args.preprocessing_workers, remove_columns=["text"]
    )

    logger.info("Tokenizing...")
    train_tok = dataset.map(tokenize, **tok_kwargs)
    eval_tok = eval_dataset.map(tokenize, **tok_kwargs) if eval_dataset else None

    logger.info(f"Grouping into blocks of {block_size}...")
    grp_kwargs = dict(batched=True, num_proc=args.preprocessing_workers)
    train_tok = train_tok.map(group_texts, **grp_kwargs)
    eval_tok = eval_tok.map(group_texts, **grp_kwargs) if eval_tok else None

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    logger.info("Loaded MaskedLM model")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=getattr(args, "mlm_probability", 0.15),
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval=eval_tok is not None),
        data_collator=collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
    )

    logger.info("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    save_outputs(trainer, tokenizer, args)
    logger.info("MLM (DAPT) complete")


# =============================================================================
# TASK: SEQUENCE CLASSIFICATION
# =============================================================================


def run_classification(args: SimpleNamespace) -> None:
    """
    Sequence-level classification.
    Dataset: parquet with 'text' + classification_column columns.
    """
    logger.info("=== Task: Sequence Classification ===")

    col = getattr(args, "classification_column", "label")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, eval_dataset = load_and_split(args)

    # Build label mapping
    unique_labels = sorted(dataset.unique(col))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Labels ({len(unique_labels)}): {unique_labels}")

    # Drop irrelevant columns
    keep = {"text", col}
    for ds_ref in [dataset, eval_dataset]:
        if ds_ref is not None:
            to_drop = [c for c in ds_ref.column_names if c not in keep]
            if to_drop:
                ds_ref = ds_ref.remove_columns(to_drop)

    def map_labels(example):
        example["labels"] = label2id[example[col]]
        return example

    dataset = dataset.map(map_labels, remove_columns=[col])
    eval_dataset = (
        eval_dataset.map(map_labels, remove_columns=[col]) if eval_dataset else None
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding=False
        )

    tok_kwargs = dict(
        batched=True, num_proc=args.preprocessing_workers, remove_columns=["text"]
    )
    train_tok = dataset.map(tokenize, **tok_kwargs)
    eval_tok = eval_dataset.map(tokenize, **tok_kwargs) if eval_dataset else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    logger.info(f"Loaded SequenceClassification model with {len(label2id)} labels")

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval=eval_tok is not None),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=compute_metrics,
    )

    logger.info("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    save_outputs(
        trainer,
        tokenizer,
        args,
        extra={"label_mapping.json": {"label2id": label2id, "id2label": id2label}},
    )
    logger.info("Classification complete")


# =============================================================================
# TASK: TOKEN CLASSIFICATION (NER)
# =============================================================================


def run_ner(args: SimpleNamespace) -> None:
    """
    Token-level classification (NER).

    Dataset: parquet with two columns:
      - words_column  : List[str]  — pre-tokenized words
      - ner_column    : List[str]  — matching string labels per word

    Labels are defined in args.labels (list of strings).
    Continuation subword tokens are masked with -100 and ignored in loss.
    """
    logger.info("=== Task: NER (Token Classification) ===")

    words_col = getattr(args, "words_column", "words")
    ner_col = getattr(args, "ner_column", "ner_tags")

    # Build label maps from config — deterministic, not inferred from data
    labels: list = args.labels
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    logger.info(f"Labels ({len(labels)}): {labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, eval_dataset = load_and_split(args)

    def tokenize_and_align(examples):
        """
        Tokenize pre-split words and align word-level labels to subword tokens.
        First subword of each word → real label
        Continuation subwords    → -100 (ignored in loss)
        Special tokens (CLS/SEP) → -100
        """
        tokenized = tokenizer(
            examples[words_col],
            is_split_into_words=True,  # input is already word-split
            truncation=True,
            max_length=512,
            padding=False,
        )

        all_labels = []
        for batch_idx, word_labels in enumerate(examples[ner_col]):
            word_ids = tokenized.word_ids(batch_index=batch_idx)
            aligned = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    # CLS / SEP token
                    aligned.append(-100)
                elif word_id != prev_word_id:
                    # First subword of a new word → assign real label
                    label_str = word_labels[word_id]
                    aligned.append(label2id.get(label_str, label2id["O"]))
                else:
                    # Continuation subword → ignore
                    aligned.append(-100)
                prev_word_id = word_id

            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    remove_cols = [words_col, ner_col]
    tok_kwargs = dict(
        batched=True, num_proc=args.preprocessing_workers, remove_columns=remove_cols
    )

    logger.info("Tokenizing and aligning labels...")
    train_tok = dataset.map(tokenize_and_align, **tok_kwargs)
    eval_tok = (
        eval_dataset.map(tokenize_and_align, **tok_kwargs) if eval_dataset else None
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # safe when loading DAPT weights into NER head
    )
    logger.info(f"Loaded TokenClassification model with {len(labels)} labels")

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Strip -100 padding, convert ids back to label strings
        true_labels = [
            [id2label[l] for l in label_row if l != -100] for label_row in label_ids
        ]
        true_preds = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, label_ids)
        ]

        result = seqeval.compute(predictions=true_preds, references=true_labels)

        # Per-label F1 breakdown
        per_label = {
            entity: scores["f1"]
            for entity, scores in result.items()
            if isinstance(scores, dict) and "f1" in scores
        }

        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
            **{f"f1_{k}": v for k, v in per_label.items()},
        }

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval=eval_tok is not None),
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=compute_metrics,
    )

    logger.info("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    save_outputs(
        trainer,
        tokenizer,
        args,
        extra={"label_mapping.json": {"label2id": label2id, "id2label": id2label}},
    )
    logger.info("NER complete")


# =============================================================================
# DISPATCH
# =============================================================================

TASK_RUNNERS = {
    "mlm": run_mlm,
    "classification": run_classification,
    "ner": run_ner,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train distilbert: mlm | classification | ner"
    )
    parser.add_argument(
        "task",
        choices=list(TASK_RUNNERS.keys()),
        help="Training task to run",
    )
    cli = parser.parse_args()

    maybe_login()

    args = load_configs(cli.task)
    if args is None:
        logger.info("Config files created. Edit them and rerun.")
        return

    TASK_RUNNERS[cli.task](args)


if __name__ == "__main__":
    main()
