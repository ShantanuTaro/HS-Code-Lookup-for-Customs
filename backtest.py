"""Backtest classification against held-out CROSS rulings and calibrate the confidence gate.

This is the kill switch for the whole product. Competitors publicly claim 92-96% at the 6-digit
level; if this number comes back near 80% the thesis is wrong, and the point of running it in week
three is to find that out in week three.

The held-out rulings are removed from the retrieval index before anything is classified. Leaving
them in would let the system retrieve the answer key and report an accuracy that does not exist.
"""
import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient

from classify import ANSWER_CONFIDENCE_THRESHOLD, BASELINE_MARKER, Classification, classify, default_chat_client
from cross import Ruling, load_rulings
from retrieve import index

RESULTS_FILE = Path(__file__).resolve().parent / "data" / "backtest.jsonl"
SWEEP_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


class CaseResult(BaseModel):
    """One held-out ruling scored against the classification the system produced for it."""

    ruling_number: str
    subject: str
    expected_hs6: str
    predicted_hs6: str | None
    confidence: float
    disposition: str
    reason: str
    citations: list[str]
    correct_hs6: bool
    correct_hs4: bool
    from_baseline: bool
    retrieved_hs6: list[str]


class Report(BaseModel):
    """Ungated accuracy, gated behaviour at the shipped threshold, and the calibration sweep."""

    corpus_size: int
    indexed: int
    cases: int
    model: str
    fell_back_to_baseline: int
    accuracy_hs6: float
    accuracy_hs6_model_only: float
    accuracy_hs4: float
    retrieval_recall_at_k: float
    threshold: float
    coverage: float
    accuracy_when_answered: float
    accuracy_when_escalated: float
    sweep: list[dict]


def split(rulings: list[Ruling], sample_size: int, seed: int) -> tuple[list[Ruling], list[Ruling]]:
    """Draw a seeded held-out sample and return (corpus, held_out), sorted for reproducibility."""
    ordered = sorted(rulings, key=lambda ruling: ruling.ruling_number)
    held_out = random.Random(seed).sample(ordered, min(sample_size, len(ordered)))
    held_out_numbers = {ruling.ruling_number for ruling in held_out}
    return [ruling for ruling in ordered if ruling.ruling_number not in held_out_numbers], held_out


def score(ruling: Ruling, result: Classification) -> CaseResult:
    """Score one classification, counting a hit against any code the ruling actually assigned."""
    expected_hs6 = {code[:6] for code in ruling.hts_codes}
    return CaseResult(
        ruling_number=ruling.ruling_number,
        subject=ruling.subject,
        expected_hs6=ruling.hs6,
        predicted_hs6=result.hs6,
        confidence=result.confidence,
        disposition=result.disposition,
        reason=result.reason,
        citations=result.citations,
        correct_hs6=result.hs6 in expected_hs6,
        from_baseline=result.reasoning.startswith(BASELINE_MARKER),
        correct_hs4=bool(result.hs6) and result.hs6[:4] in {code[:4] for code in expected_hs6},
        retrieved_hs6=[hit.hs6 for hit in result.candidates],
    )


def sweep(results: list[CaseResult]) -> list[dict]:
    """Report coverage and precision at each candidate threshold, so the constant is chosen by data.

    Coverage is the share of queries answered instead of escalated; precision is accuracy among
    those answers. The right threshold is the lowest one whose precision still clears the bar an
    entry filing needs, which is what makes this a calibration rather than a preference.
    """
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        answered = [result for result in results if result.confidence >= threshold and result.predicted_hs6]
        rows.append(
            {
                "threshold": threshold,
                "coverage": len(answered) / len(results) if results else 0.0,
                "precision_hs6": sum(result.correct_hs6 for result in answered) / len(answered) if answered else 0.0,
                "precision_hs4": sum(result.correct_hs4 for result in answered) / len(answered) if answered else 0.0,
                "answered": len(answered),
            }
        )
    return rows


def run(sample_size: int, seed: int, workers: int, use_model: bool) -> Report:
    """Split, index the remainder, classify every held-out ruling, and summarize."""
    rulings = load_rulings()
    if len(rulings) < sample_size * 2:
        raise SystemExit(f"Only {len(rulings)} rulings in the corpus; run cross.py before backtesting.")
    corpus, held_out = split(rulings, sample_size, seed)
    client = QdrantClient(":memory:")
    indexed = index(client, corpus)
    chat = default_chat_client() if use_model else None

    # Results are written as they land, and progress is printed, because the free tier's token
    # budget makes this a multi-hour run: a crash at case 400 must not throw away 400 classifications,
    # and a run with no output is indistinguishable from a hung one.
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool, RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for result in pool.map(lambda ruling: score(ruling, classify(ruling.description, client=client, chat=chat)), held_out):
            results.append(result)
            handle.write(result.model_dump_json() + "\n")
            handle.flush()
            done = len(results)
            if done % 10 == 0 or done == len(held_out):
                rate = done / max(time.monotonic() - started, 1e-9)
                correct = sum(item.correct_hs6 for item in results)
                print(f"  {done}/{len(held_out)} | HS6 {correct / done:.1%} | {rate * 60:.1f}/min | ~{(len(held_out) - done) / max(rate, 1e-9) / 60:.0f} min left", file=sys.stderr, flush=True)

    from_model = [result for result in results if not result.from_baseline]
    answered = [result for result in results if result.disposition == "answered"]
    escalated = [result for result in results if result.disposition == "escalated"]
    return Report(
        corpus_size=len(rulings),
        indexed=indexed,
        cases=len(results),
        model=(chat.model if chat else "retrieval-baseline"),
        fell_back_to_baseline=sum(result.from_baseline for result in results),
        accuracy_hs6=sum(result.correct_hs6 for result in results) / len(results),
        # Reported separately because a rate-limited call falls back to the baseline, and averaging
        # the two would quietly report the baseline's accuracy as the model's.
        accuracy_hs6_model_only=(sum(r.correct_hs6 for r in from_model) / len(from_model)) if from_model else 0.0,
        accuracy_hs4=sum(result.correct_hs4 for result in results) / len(results),
        retrieval_recall_at_k=sum(result.expected_hs6 in result.retrieved_hs6 for result in results) / len(results),
        threshold=ANSWER_CONFIDENCE_THRESHOLD,
        coverage=len(answered) / len(results),
        accuracy_when_answered=sum(result.correct_hs6 for result in answered) / len(answered) if answered else 0.0,
        accuracy_when_escalated=sum(result.correct_hs6 for result in escalated) / len(escalated) if escalated else 0.0,
        sweep=sweep(results),
    )


def main() -> None:
    """Run the held-out backtest and print the report plus the threshold calibration table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-model", action="store_true", help="Score the retrieval baseline instead of the model.")
    args = parser.parse_args()
    report = run(args.cases, args.seed, args.workers, not args.no_model)
    print(report.model_dump_json(indent=2, exclude={"sweep"}))
    print(f"\n{'threshold':>10}{'answered':>10}{'coverage':>10}{'HS6 prec':>10}{'HS4 prec':>10}")
    for row in report.sweep:
        print(f"{row['threshold']:>10.2f}{row['answered']:>10}{row['coverage']:>10.1%}{row['precision_hs6']:>10.1%}{row['precision_hs4']:>10.1%}")


if __name__ == "__main__":
    main()
