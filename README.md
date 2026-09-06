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
| `app.py`       | Lookup page, 24k indexable ruling pages, sitemap, JSON endpoint            |
| `test_hs.py`   | Leakage, gate, retrieval and split checks                                   |

### Label leakage

The backtest is worthless if the eval input contains its own answer. `cross.extract_description`
cuts each ruling at the first conclusion marker *and* at the first HTS code, which also removes
subheadings the importer proposed and CBP rejected. `backtest.split` removes the held-out rulings
from the index before anything is classified. Both are asserted in `test_hs.py`.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY and/or MISTRAL_API_KEY

.venv/bin/python cross.py --from-year 2015 --to-year 2026   # ~30k rulings, resumable
.venv/bin/python backtest.py --cases 500                    # the kill switch
.venv/bin/python backtest.py --cases 500 --no-model         # retrieval-only floor
.venv/bin/python -m pytest test_hs.py -q

.venv/bin/uvicorn app:app --port 8099   # the free lookup page
```

## The SEO surface

The lookup page is one URL competing against Flexport and Avalara. The corpus is 23,929 URLs
competing against nothing: `/ruling/N352926` is a real page about a real product, titled with the
long-tail phrase someone actually searches ("tariff classification of a hat, a headband and a
blanket from China"). Each one links to the six rulings nearest it in retrieval space, so the corpus
is a connected graph rather than 23,929 orphans, and each links into the lookup tool.

`/sitemap.xml` lists every page; `robots.txt` points at it and keeps `/?q=` lookups out of the
index, since a separate indexable page per query is how a tool becomes thin content. Set `BASE_URL`
in the deployment or every canonical tag will claim the pages live on localhost.

### Providers

Both Groq and Mistral speak the OpenAI chat-completions dialect, so one client covers them and the
only difference is a base URL. Every key configured in `.env` becomes a link in a fallback chain:
all but the last give up after two retries, so a rate-limited provider costs seconds rather than the
run. Which model answered is stamped on every result and reported per-provider, because a chain is
two different models and one blended accuracy number would hide which is doing the work.

`backtest.py` prints ungated HS6/HS4 accuracy, retrieval recall@k (the ceiling on everything else),
and a coverage-vs-precision table across candidate thresholds. Pick the lowest threshold whose
precision clears what an entry filing needs; that number goes in `classify.ANSWER_CONFIDENCE_THRESHOLD`.

## Status

- [x] 1. Repoint retrieval at CROSS
- [ ] 2. Backtest 500 published rulings — **the kill switch**; near 80% means the thesis is wrong
- [x] 3. Free single-lookup page (SEO surface)
- [ ] 4. Stripe-billed API tier
