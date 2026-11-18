"""Raw news text to structured PERSON, ORGANIZATION, LOCATION entities."""

import argparse
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_common import ROOT, load_conll, spans


TYPE_NAMES = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}


class Extractor:
    def __init__(self, checkpoint=ROOT / "checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            checkpoint, local_files_only=True).to(self.device).eval()

    def extract(self, text):
        encoded = self.tokenizer(text, return_offsets_mapping=True,
                                 return_tensors="pt", truncation=True,
                                 max_length=256)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = self.model(**{key: value.to(self.device)
                                   for key, value in encoded.items()}).logits[0]
        labels = [self.model.config.id2label[int(index)]
                  for index in logits.argmax(-1).cpu().tolist()]
        entities = []
        active = None
        for (start, end), label in zip(offsets, labels):
            if start == end:
                continue
            if label == "O" or label.split("-", 1)[-1] not in TYPE_NAMES:
                if active is not None:
                    entities.append(active)
                    active = None
                continue
            prefix, kind = label.split("-", 1)
            if active is not None and prefix == "I" and kind == active[2]:
                active = (active[0], end, kind)
            else:
                if active is not None:
                    entities.append(active)
                active = (start, end, kind)
        if active is not None:
            entities.append(active)
        output = {name: [] for name in TYPE_NAMES.values()}
        for start, end, kind in entities:
            output[TYPE_NAMES[kind]].append(text[start:end])
        return output, entities


def gold_char_spans(tokens, labels):
    starts = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token) + 1
    return {(starts[start], starts[end - 1] + len(tokens[end - 1]), kind)
            for start, end, kind in spans(labels) if kind in TYPE_NAMES}


def evaluate(extractor, count=30):
    data, names = load_conll()
    selected = data["test"].select(range(count))
    tp = fp = fn = 0
    examples = []
    for row in selected:
        text = " ".join(row["tokens"])
        structured, predicted = extractor.extract(text)
        gold = gold_char_spans(row["tokens"],
                               [names[i] for i in row["ner_tags"]])
        pred = set(predicted)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        examples.append({"text": text, "output": structured,
                         "gold": sorted(gold), "prediction": predicted})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"sentences": count, "tp": tp, "fp": fp, "fn": fn,
               "precision": precision, "recall": recall, "f1": f1,
               "match": "exact character span and type"}
    (ROOT / "ie_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "ie_examples.json").write_text(json.dumps(examples, ensure_ascii=False,
                                                       indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    args = parser.parse_args()
    extractor = Extractor()
    if args.text:
        print(json.dumps(extractor.extract(args.text)[0], ensure_ascii=False,
                         indent=2))
    else:
        evaluate(extractor)
