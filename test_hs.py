"""The checks that matter: label leakage, the gate, and retrieval doing what the backtest assumes."""
from qdrant_client import QdrantClient

import backtest
import classify
import cross
import retrieve

RULING_TEXT = """N123456\r\rMay 1, 2024\r\rCATEGORY:\tClassification\r\rTARIFF NO.: 6109.10.0012\r\r
Ms. Jane Doe\rAcme Imports\r\rRE:\tThe tariff classification of a cotton t-shirt from Vietnam\r\r
Dear Ms. Doe:\r\rIn your letter dated April 1, 2024, you requested a tariff classification ruling.\r\r
The submitted sample is a men's t-shirt constructed from 100% cotton jersey knit fabric weighing 180 grams per square meter. \
The garment features short hemmed sleeves, a rib knit crew neckline and a straight hemmed bottom. It is imported in sizes small through extra large.\r\r
You suggested classification under 6110.20.2079, which provides for sweaters and pullovers. We disagree.\r\r
The applicable subheading for the t-shirt will be 6109.10.0012, HTSUS, which provides for T-shirts, knitted, of cotton. The rate of duty will be 16.5 percent.\r\r
Sincerely,\rDirector\r"""


def test_description_never_leaks_a_code_or_the_holding():
    description = cross.extract_description(RULING_TEXT)
    assert "cotton jersey knit" in description
    assert not cross.HTS_CODE.findall(description), "an HTS code survived into the eval input"
    assert "applicable subheading" not in description
    assert "6110.20" not in description, "the rejected subheading the importer proposed also leaks the answer space"


def test_codes_normalize_and_drop_fragments():
    assert cross.normalize_codes("6109.10.0012, 6110.20.2079") == ["6109100012", "6110202079"]
    assert cross.normalize_codes("Chapter 61") == []
    assert cross.normalize_codes("") == []


def make_ruling(number: str, code: str, subject: str, description: str) -> cross.Ruling:
    return cross.Ruling(ruling_number=number, collection="ny", date="2024-01-01", subject=subject, hts_codes=[code], description=description, text="")


def test_retrieval_finds_the_matching_ruling():
    client = QdrantClient(":memory:")
    retrieve.index(
        client,
        [
            make_ruling("N1", "6109100012", "cotton t-shirt", "a men's t-shirt of 100% cotton jersey knit fabric"),
            make_ruling("N2", "8471300100", "laptop computer", "a portable automatic data processing machine with a keyboard"),
            make_ruling("N3", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated"),
        ],
    )
    assert retrieve.search("knit cotton t-shirt for men", top_k=1, client=client)[0].ruling_number == "N1"
    assert retrieve.search("portable computer with keyboard", top_k=1, client=client)[0].ruling_number == "N2"


def make_result(**overrides) -> classify.Classification:
    hits = [retrieve.Hit(ruling_number="N1", hs6="610910", hts_codes=["6109100012"], subject="s", date="2024-01-01", description="d", score=1.0, rerank_score=1.0)]
    fields = {"hs6": "610910", "reasoning": "r", "citations": ["N1"], "confidence": 0.95, "disposition": "escalated", "reason": "", "candidates": hits}
    return classify.Classification(**{**fields, **overrides})


def test_gate_answers_only_supported_confident_classifications():
    assert classify.gate(make_result()).disposition == "answered"
    assert classify.gate(make_result(confidence=0.5)).disposition == "escalated"
    assert classify.gate(make_result(citations=[])).disposition == "escalated"
    assert classify.gate(make_result(hs6=None)).disposition == "escalated"
    assert classify.gate(make_result(hs6="61")).disposition == "escalated"
    invented = classify.gate(make_result(citations=["N1", "N999999"]))
    assert invented.disposition == "escalated" and "N999999" in invented.reason, "a hallucinated citation must never be answerable"
    assert classify.gate(make_result()).answer == "610910"
    assert classify.gate(make_result(confidence=0.5)).answer is None


def test_holdout_is_removed_from_the_corpus():
    rulings = [make_ruling(f"N{i}", "6109100012", "s", "d") for i in range(20)]
    corpus, held_out = backtest.split(rulings, sample_size=5, seed=1)
    assert len(held_out) == 5 and len(corpus) == 15
    assert not {r.ruling_number for r in corpus} & {r.ruling_number for r in held_out}
    assert backtest.split(rulings, 5, 1)[1] == held_out, "the split must be reproducible across runs"
