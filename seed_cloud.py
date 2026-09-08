"""Seed a hosted Qdrant with the CROSS corpus, from a machine that has `data/rulings.jsonl`.

The deployed app never does this: the corpus file is 790MB and a serverless start has neither the
disk nor the minutes. Point ids are derived from the ruling number, so a run that dies part-way is
resumed by running it again - the upserts that already landed are simply rewritten.

    python seed_cloud.py            # credentials from .env.cloud
"""
import sys

from dotenv import dotenv_values
from qdrant_client import QdrantClient

from cross import RULINGS_FILE, load_rulings
from retrieve import COLLECTION, index


def ticking(rulings, every=5_000):
    """Pass rulings through, reporting progress: the run takes tens of minutes over the network."""
    for number, ruling in enumerate(rulings, 1):
        if number % every == 0:
            print(f"{number:,} sent", flush=True)
        yield ruling


def main() -> int:
    config = dotenv_values(".env.cloud")
    if not config.get("QDRANT_URL"):
        sys.exit("No QDRANT_URL in .env.cloud")
    client = QdrantClient(url=config["QDRANT_URL"], api_key=config.get("QDRANT_API_KEY"), timeout=300)
    rulings = load_rulings(RULINGS_FILE)
    print(f"{len(rulings):,} rulings loaded, seeding {config['QDRANT_URL']}", flush=True)
    index(client, ticking(rulings))
    total = client.count(collection_name=COLLECTION).count
    print(f"done: {total:,} points in {COLLECTION}", flush=True)
    return total


if __name__ == "__main__":
    main()
