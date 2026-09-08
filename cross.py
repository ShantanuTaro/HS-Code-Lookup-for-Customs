"""Fetch CBP CROSS rulings and split each into an importer-style description and its HTS label.

CROSS is the whole wedge: 200k+ public classification rulings that are simultaneously the retrieval
corpus and the eval ground truth. This module is the only place that talks to rulings.cbp.gov.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

BASE_URL = "https://rulings.cbp.gov/api"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
RULINGS_FILE = DATA_DIRECTORY / "rulings.jsonl"

# `term` is required by the API but stopwords match every document, so a stopword plus a date
# window enumerates a period completely instead of sampling it. Windows stay under the API's
# 10,000-hit ceiling; a year of rulings is 2-5k, so one request window per year is safe.
ENUMERATION_TERM = "a"
MAX_PAGE_SIZE = 500

HTS_CODE = re.compile(r"\b\d{4}\.\d{2}[.\d]*\b")
# The first line that states a conclusion ends the description. Everything after it is the answer,
# so leaving any of it in would leak the label into the eval input.
HOLDING_MARKERS = re.compile(
    r"applicable subheading|is classified|are classified|classifiable|is provided for|"
    r"^\s*HOLDING|^\s*ISSUE|^\s*LAW AND ANALYSIS|we find that|it is the decision",
    re.IGNORECASE | re.MULTILINE,
)
# Not anchored to a line: rulings issued from 2025 on flow the salutation and the first body
# sentence onto one line, and an anchored match silently found nothing there, leaving the
# TARIFF NO. header in the text so the description was cut to nothing a hundred characters later.
SALUTATION = re.compile(r"Dear\s+[^\n:]{0,80}:")
MINIMUM_DESCRIPTION_CHARACTERS = 200
# CBP throttles after a few thousand requests. Without a retry the crawl keeps running and simply
# stops collecting, which is worse than stopping: it produces a corpus with a hole in its most
# recent years and no indication that anything went wrong.
FETCH_RETRIES = 5


class Ruling(BaseModel):
    """One CROSS ruling reduced to what classification needs: a description and its codes."""

    ruling_number: str
    collection: str
    date: str
    subject: str
    hts_codes: list[str] = Field(min_length=1)
    description: str
    text: str

    @property
    def hs6(self) -> str:
        """The 6-digit international subheading, which is the level accuracy is claimed at."""
        return self.hts_codes[0][:6]


def normalize_codes(tariffs: str) -> list[str]:
    """Return the substantive HTS codes, dropping Chapter 98 and 99 provisions.

    Chapters 98 and 99 are US-only: special classification provisions and trade remedies such as
    Section 301 and IEEPA duties. They ride alongside the real subheading rather than replacing it,
    and they do not exist in the 6-digit international HS at all, so scoring a prediction against
    `990301` would be scoring it against something no classifier should ever return.
    """
    seen: dict[str, None] = {}
    for candidate in re.findall(r"\d{4}[.\d]*", tariffs or ""):
        digits = candidate.replace(".", "")
        if len(digits) >= 6 and digits[:2] not in {"98", "99"}:
            seen.setdefault(digits, None)
    return list(seen)


def extract_description(text: str) -> str:
    """Return the merchandise description with every trace of the ruling's conclusion removed.

    Two independent cuts, because either alone leaks. The holding marker catches the prose that
    announces the decision; the code pattern catches a subheading quoted earlier, including one the
    importer proposed and CBP rejected. A real caller has neither, so neither may reach the model.
    """
    body = text.replace("\r", "\n")
    salutation = SALUTATION.search(body)
    if salutation:
        body = body[salutation.end():]
    cuts = [match.start() for match in (HOLDING_MARKERS.search(body), HTS_CODE.search(body)) if match]
    if cuts:
        body = body[: min(cuts)]
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def search_year(client: httpx.Client, year: int) -> list[dict]:
    """Enumerate one year of classification rulings, following pagination to the end."""
    rows: list[dict] = []
    for page in range(1, 100):
        response = client.get(
            f"{BASE_URL}/search",
            params={
                "term": ENUMERATION_TERM,
                "collection": "ALL",
                "pageSize": MAX_PAGE_SIZE,
                "page": page,
                "sortBy": "DATE",
                "fromDate": f"{year}-01-01",
                "toDate": f"{year}-12-31",
            },
        )
        response.raise_for_status()
        batch = response.json()["rulings"]
        rows.extend(batch)
        if len(batch) < MAX_PAGE_SIZE:
            break
    return [row for row in rows if row.get("categories") == "Classification" and normalize_codes(row.get("tariffs", ""))]


def fetch_ruling(client: httpx.Client, row: dict) -> Ruling | None:
    """Fetch one ruling and build the record, retrying transport failures before giving up.

    Returns `None` only for a ruling that is genuinely unusable - one whose text carries no
    description once the holding is removed. A fetch that never succeeded raises instead, so a
    throttled crawl is visible as an error rather than as a smaller corpus.
    """
    text = None
    for attempt in range(FETCH_RETRIES):
        try:
            response = client.get(f"{BASE_URL}/ruling/{row['rulingNumber']}")
            response.raise_for_status()
            text = response.json()["text"]
            break
        except (httpx.HTTPError, LookupError, ValueError):
            if attempt == FETCH_RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    description = extract_description(text or "")
    if len(description) < MINIMUM_DESCRIPTION_CHARACTERS:
        return None
    return Ruling(
        ruling_number=row["rulingNumber"],
        collection=row["collection"],
        date=row["rulingDate"][:10],
        subject=row["subject"],
        hts_codes=normalize_codes(row["tariffs"]),
        description=description,
        text=text,
    )


def load_rulings(path: Path = RULINGS_FILE) -> list[Ruling]:
    """Read the local corpus, tolerating a partially written final line from an interrupted crawl."""
    if not path.exists():
        return []
    rulings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rulings.append(Ruling.model_validate_json(line))
        except ValueError:
            continue
    return rulings


def reparse(path: Path = RULINGS_FILE) -> tuple[int, int]:
    """Rebuild every stored record's description and codes from its saved text, in place.

    The full ruling text is kept precisely so that a parser fix does not require re-crawling 20,000
    rulings from CBP. Returns (kept, dropped).
    """
    rulings = load_rulings(path)
    rebuilt: list[Ruling] = []
    for ruling in rulings:
        codes = [code for code in ruling.hts_codes if code[:2] not in {"98", "99"}]
        description = extract_description(ruling.text) if ruling.text else ruling.description
        if codes and len(description) >= MINIMUM_DESCRIPTION_CHARACTERS:
            rebuilt.append(ruling.model_copy(update={"hts_codes": codes, "description": description}))
    path.write_text("".join(ruling.model_dump_json() + "\n" for ruling in rebuilt), encoding="utf-8")
    return len(rebuilt), len(rulings) - len(rebuilt)


def crawl(from_year: int, to_year: int, workers: int = 8, path: Path = RULINGS_FILE) -> int:
    """Append every not-yet-fetched ruling in the year range, so an interrupted crawl resumes cheaply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    have = {ruling.ruling_number for ruling in load_rulings(path)}
    added = 0
    with httpx.Client(timeout=60.0, headers={"User-Agent": "hs-classify/0.1"}) as client:
        for year in range(from_year, to_year + 1):
            rows = [row for row in search_year(client, year) if row["rulingNumber"] not in have]
            failed = unusable = 0

            def fetch(row: dict) -> Ruling | None:
                nonlocal failed
                try:
                    return fetch_ruling(client, row)
                except (httpx.HTTPError, LookupError, ValueError):
                    failed += 1
                    return None

            with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
                for ruling in pool.map(fetch, rows):
                    if ruling is None:
                        unusable += 1
                    else:
                        handle.write(ruling.model_dump_json() + "\n")
                        added += 1
                handle.flush()
            print(f"{year}: {len(rows)} found, {len(rows) - failed - unusable} written, {unusable} unusable, {failed} FAILED ({added} total)", flush=True)
            time.sleep(1)
    return added


def main() -> None:
    """Crawl a year range of CROSS classification rulings into the local corpus file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-year", type=int, default=2015)
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reparse", action="store_true", help="Rebuild descriptions from stored text instead of crawling.")
    args = parser.parse_args()
    if args.reparse:
        kept, dropped = reparse()
        print(f"reparsed {kept} rulings, dropped {dropped}")
        return
    print(f"wrote {crawl(args.from_year, args.to_year, args.workers)} rulings to {RULINGS_FILE}")


if __name__ == "__main__":
    main()
