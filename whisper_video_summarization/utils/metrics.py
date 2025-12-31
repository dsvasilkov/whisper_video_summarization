from typing import List
from rouge_score import rouge_scorer


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}
