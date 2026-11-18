"""Fine-tune and evaluate a small pretrained BERT token classifier."""

import json

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification)

from ner_common import (MODEL, ROOT, align_labels, load_conll,
                        save_predictions, score_entities, seed_everything)


def encode_split(split, tokenizer):
    def encode(batch):
        encodings = tokenizer(batch["tokens"], is_split_into_words=True,
                              truncation=True, max_length=256)
        encodings["labels"] = [align_labels(encodings[i], labels)
                               for i, labels in enumerate(batch["ner_tags"])]
        return encodings

    return split.map(encode, batched=True, remove_columns=split.column_names)


def evaluate(model, encoded, raw, names, collator, device):
    loader = DataLoader(encoded, batch_size=32, collate_fn=collator)
    predictions = []
    gold_flat = []
    pred_flat = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            predicted = model(**inputs).logits.argmax(-1).cpu()
            for gold_ids, pred_ids in zip(labels.tolist(), predicted.tolist()):
                valid = [(gold, pred) for gold, pred in zip(gold_ids, pred_ids)
                         if gold != -100]
                gold_flat.extend(g for g, _ in valid)
                pred_flat.extend(p for _, p in valid)
                predictions.append([names[p] for _, p in valid])
    if any(len(pred) != len(row["tokens"])
           for pred, row in zip(predictions, raw)):
        raise ValueError("A sentence was truncated; word-level evaluation invalid")
    gold = [[names[i] for i in row["ner_tags"]] for row in raw]
    metrics = score_entities(gold, predictions)
    token_p, token_r, token_f1, _ = precision_recall_fscore_support(
        gold_flat, pred_flat, labels=list(range(1, len(names))),
        average="micro", zero_division=0)
    metrics["token_precision"] = token_p
    metrics["token_recall"] = token_r
    metrics["token_f1"] = token_f1
    metrics["token_accuracy"] = float(np.mean(np.array(gold_flat) == np.array(pred_flat)))
    return metrics, predictions


def run(epochs=10):
    seed_everything()
    data, names = load_conll()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoded = {split: encode_split(data[split], tokenizer)
               for split in ("train", "validation", "test")}
    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(encoded["train"], batch_size=32, shuffle=True,
                        collate_fn=collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(names), id2label=dict(enumerate(names)),
        label2id={name: index for index, name in enumerate(names)}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    best_f1 = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        validation, _ = evaluate(model, encoded["validation"],
                                 data["validation"], names, collator, device)
        history.append({"epoch": epoch + 1, "train_loss": running_loss / len(loader),
                        "validation_entity_f1": validation["f1"]})
        print(json.dumps(history[-1]), flush=True)
        if validation["f1"] > best_f1:
            best_f1 = validation["f1"]
            model.save_pretrained(ROOT / "checkpoint")
            tokenizer.save_pretrained(ROOT / "checkpoint")
    model = AutoModelForTokenClassification.from_pretrained(
        ROOT / "checkpoint", local_files_only=True).to(device)
    metrics, predictions = evaluate(model, encoded["test"], data["test"],
                                    names, collator, device)
    save_predictions(ROOT / "transformer_predictions.json", data["test"],
                     names, predictions)
    (ROOT / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "training_history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"test": metrics}, indent=2))
    return metrics


if __name__ == "__main__":
    run()
