# hs-classify

HS code classification traced to CBP rulings, behind a confidence gate. Sold as a self-serve USD API.

## Why this exists

A wrong HS code on an entry is a false statement to CBP under 19 U.S.C. §1592. Competitors publish
92–96% accuracy at the 6-digit level and stop there. Nobody sells a classification that is *legally
defensible*: reasoning traced to specific published rulings, with a calibrated gate that declines to
answer rather than guessing. The gate is the product, not a safety feature bolted onto it.

The wedge is CBP CROSS: 200,000+ public rulings, free, deep, owned by nobody, and usable as both the
retrieval corpus and the eval ground truth. Being able to *measure* is the moat.

## Architecture

Transferred node-for-node from the freight exception agent, pointed at a problem where being wrong
has a statutory citation attached.

| Freight agent          | Here                                    |
| ---------------------- | --------------------------------------- |
| `classify_exception`   | `classify.classify`                     |
| `retrieve_playbook`    | `retrieve.search` over CROSS rulings    |
| `decide_authorization` | `classify.gate` — answer vs. escalate   |

Same machinery: a deterministic gate the model cannot talk its way past, one auto-answer threshold
in one place, a labelled fallback instead of a silent one, and a threshold calibrated against
measured accuracy rather than taste.

## Files

| File           | Does                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| `cross.py`     | Crawl rulings.cbp.gov; split each ruling into description + HTS label       |
| `retrieve.py`  | Hybrid dense+sparse retrieval over the corpus, RRF-fused and reranked       |
| `classify.py`  | Classify against retrieved rulings, then apply the gate                     |
| `backtest.py`  | Held-out accuracy and the threshold calibration sweep                       |
| `test_hs.py`   | Leakage, gate, retrieval and split checks                                   |

### Label leakage

The backtest is worthless if the eval input contains its own answer. `cross.extract_description`
cuts each ruling at the first conclusion marker *and* at the first HTS code, which also removes
subheadings the importer proposed and CBP rejected. `backtest.split` removes the held-out rulings
from the index before anything is classified. Both are asserted in `test_hs.py`.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY

.venv/bin/python cross.py --from-year 2015 --to-year 2026   # ~30k rulings, resumable
.venv/bin/python backtest.py --cases 500                    # the kill switch
.venv/bin/python backtest.py --cases 500 --no-model         # retrieval-only floor
.venv/bin/python -m pytest test_hs.py -q
```

`backtest.py` prints ungated HS6/HS4 accuracy, retrieval recall@k (the ceiling on everything else),
and a coverage-vs-precision table across candidate thresholds. Pick the lowest threshold whose
precision clears what an entry filing needs; that number goes in `classify.ANSWER_CONFIDENCE_THRESHOLD`.

## Status

- [x] 1. Repoint retrieval at CROSS
- [ ] 2. Backtest 500 published rulings — **the kill switch**; near 80% means the thesis is wrong
- [ ] 3. Free single-lookup page (SEO surface)
- [ ] 4. Stripe-billed API tier
