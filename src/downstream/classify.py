"""
Downstream-классификация на эмбеддингах.
LogReg, stratified split, accuracy + ROC-AUC.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import get_logger


def prepare_labels(
    user_ids: list,
    users_df,
    task: str = "gender",
) -> Tuple[np.ndarray, LabelEncoder, np.ndarray]:
    """Подготовить метки для классификации."""
    logger = get_logger()

    user_lookup = users_df.set_index("user_id")

    labels = []
    valid_mask = []
    for uid in user_ids:
        if uid in user_lookup.index:
            row = user_lookup.loc[uid]
            if task == "gender":
                labels.append(row["gender"])
                valid_mask.append(True)
            elif task == "age_bucket":
                age = row["age"]
                labels.append("young" if age < 30 else "older")
                valid_mask.append(True)
            else:
                raise ValueError(f"Unknown task: {task}")
        else:
            labels.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    labels = [l for l, v in zip(labels, valid_mask) if v]

    le = LabelEncoder()
    encoded = le.fit_transform(labels)

    logger.info(
        f"Downstream task '{task}': {len(encoded)} samples, "
        f"classes={[str(c) for c in le.classes_]}, "
        f"distribution={np.bincount(encoded).tolist()}"
    )

    return encoded, le, valid_mask


def train_and_evaluate(
    embeddings: np.ndarray,
    user_ids: list,
    users_df,
    task: str = "gender",
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Обучить LogReg и вывести метрики."""
    logger = get_logger()

    labels, le, valid_mask = prepare_labels(user_ids, users_df, task)

    X = embeddings[valid_mask]
    y = labels

    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.warning("Only one class in dataset, cannot train.")
        return {"error": "single_class", "accuracy": 0.0, "roc_auc": None}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    class_names = [str(c) for c in le.classes_]
    train_dist = np.bincount(y_train, minlength=n_classes).tolist()
    test_dist = np.bincount(y_test, minlength=n_classes).tolist()
    logger.info(f"Train size: {len(y_train)}, distribution: {dict(zip(class_names, train_dist))}")
    logger.info(f"Test size:  {len(y_test)}, distribution: {dict(zip(class_names, test_dist))}")

    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")

    roc_auc = None
    test_classes = len(np.unique(y_test))
    if test_classes < 2:
        logger.warning("Test set has only one class, ROC-AUC undefined.")
    else:
        y_prob = clf.predict_proba(X_test)
        if len(le.classes_) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": class_names,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_distribution": dict(zip(class_names, train_dist)),
        "test_distribution": dict(zip(class_names, test_dist)),
    }
