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
SALUTATION = re.compile(r"^Dear\b.*:\s*$", re.MULTILINE)
MINIMUM_DESCRIPTION_CHARACTERS = 200


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
    """Turn the API's free-text tariff field into deduplicated digit-only HTS codes."""
    seen: dict[str, None] = {}
    for candidate in re.findall(r"\d{4}[.\d]*", tariffs or ""):
        digits = candidate.replace(".", "")
        if len(digits) >= 6:
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
    """Fetch one ruling's full text and build the record, skipping any that cannot be split cleanly."""
    try:
        response = client.get(f"{BASE_URL}/ruling/{row['rulingNumber']}")
        response.raise_for_status()
        text = response.json()["text"]
    except (httpx.HTTPError, LookupError, ValueError):
        return None
    description = extract_description(text)
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


def crawl(from_year: int, to_year: int, workers: int = 8, path: Path = RULINGS_FILE) -> int:
    """Append every not-yet-fetched ruling in the year range, so an interrupted crawl resumes cheaply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    have = {ruling.ruling_number for ruling in load_rulings(path)}
    added = 0
    with httpx.Client(timeout=60.0, headers={"User-Agent": "hs-classify/0.1"}) as client:
        for year in range(from_year, to_year + 1):
            rows = [row for row in search_year(client, year) if row["rulingNumber"] not in have]
            with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
                for ruling in pool.map(lambda row: fetch_ruling(client, row), rows):
                    if ruling is not None:
                        handle.write(ruling.model_dump_json() + "\n")
                        added += 1
                handle.flush()
            print(f"{year}: {len(rows)} new, {added} total written", flush=True)
            time.sleep(1)
    return added


def main() -> None:
    """Crawl a year range of CROSS classification rulings into the local corpus file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-year", type=int, default=2015)
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(f"wrote {crawl(args.from_year, args.to_year, args.workers)} rulings to {RULINGS_FILE}")


if __name__ == "__main__":
    main()
