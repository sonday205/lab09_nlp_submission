"""Train a token-feature Logistic Regression baseline on CoNLL-2003."""

import argparse
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ner_common import ROOT, SEED, load_conll, save_predictions, score_entities, seed_everything


def features(tokens, index):
    word = tokens[index]
    previous = tokens[index - 1] if index else "<START>"
    following = tokens[index + 1] if index + 1 < len(tokens) else "<END>"
    return {
        "word": word.lower(),
        "prefix2": word[:2].lower(),
        "prefix3": word[:3].lower(),
        "suffix2": word[-2:].lower(),
        "suffix3": word[-3:].lower(),
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "previous": previous.lower(),
        "following": following.lower(),
        "previous_title": previous.istitle(),
        "following_title": following.istitle(),
    }


def flatten_features(split):
    return [features(row["tokens"], i) for row in split for i in range(len(row["tokens"]))]


def run():
    seed_everything()
    data, names = load_conll()
    vectorizer = DictVectorizer()
    train_x = vectorizer.fit_transform(flatten_features(data["train"]))
    train_y = [tag for row in data["train"] for tag in row["ner_tags"]]
    model = LogisticRegression(max_iter=150, random_state=SEED, solver="lbfgs")
    model.fit(train_x, train_y)
    test_x = vectorizer.transform(flatten_features(data["test"]))
    flat_predictions = model.predict(test_x)
    predictions = []
    offset = 0
    for row in data["test"]:
        length = len(row["tokens"])
        predictions.append([names[i] for i in flat_predictions[offset:offset + length]])
        offset += length
    rows = save_predictions(ROOT / "baseline_predictions.json", data["test"], names, predictions)
    metrics = score_entities([r["gold"] for r in rows], predictions)
    (ROOT / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
