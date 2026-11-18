import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATASET = "tomaarsen/conll2003"
MODEL = "google/bert_uncased_L-2_H-128_A-2"
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_conll():
    data = load_dataset(DATASET)
    names = data["train"].features["ner_tags"].feature.names
    return data, names


def align_labels(encoding, word_labels):
    aligned = []
    previous = None
    word_ids = encoding.word_ids
    for word_id in word_ids() if callable(word_ids) else word_ids:
        if word_id is None or word_id == previous:
            aligned.append(-100)
        else:
            aligned.append(word_labels[word_id])
        previous = word_id
    return aligned


def spans(labels):
    found = []
    start = None
    kind = None
    for index, label in enumerate(list(labels) + ["O"]):
        if label == "O":
            if kind is not None:
                found.append((start, index, kind))
            start = kind = None
            continue
        prefix, new_kind = label.split("-", 1)
        if prefix == "B" or new_kind != kind or kind is None:
            if kind is not None:
                found.append((start, index, kind))
            start, kind = index, new_kind
    return found


def score_entities(golds, predictions, types=None):
    tp = fp = fn = 0
    for gold, predicted in zip(golds, predictions):
        gold_spans = set(spans(gold))
        pred_spans = set(spans(predicted))
        if types is not None:
            gold_spans = {s for s in gold_spans if s[2] in types}
            pred_spans = {s for s in pred_spans if s[2] in types}
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def save_predictions(path, split, names, predictions):
    rows = []
    for item, pred in zip(split, predictions):
        rows.append({"id": item["id"], "tokens": item["tokens"],
                     "gold": [names[i] for i in item["ner_tags"]],
                     "prediction": pred})
    if len(rows) != len(split):
        raise ValueError("Prediction count differs from test set size")
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows
