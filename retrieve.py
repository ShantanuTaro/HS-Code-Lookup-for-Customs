"""Hybrid retrieval over the CROSS corpus: dense + sparse fusion, then lexical rerank.

The encoders are deterministic local hashing rather than a model, so indexing 30k rulings needs no
download, no GPU and no per-token cost, and an index built today is bit-identical to one built in
CI six months from now. Swap in a trained encoder only if the backtest shows retrieval is the
bottleneck; the search contract below does not change when it does.
"""
import hashlib
import math
import re
import uuid
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Fusion, FusionQuery, PointStruct, Prefetch, SparseVector, SparseVectorParams, VectorParams

from cross import Ruling

COLLECTION = "cross_rulings"
DENSE_DIMENSIONS = 384
SPARSE_VOCABULARY_SIZE = 65_536
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MINIMUM_PREFETCH = 40
RERANK_CANDIDATES = 40


class Hit(BaseModel):
    """One retrieved ruling with its fused retrieval score and post-rerank score."""

    ruling_number: str
    hs6: str
    hts_codes: list[str]
    subject: str
    date: str
    description: str
    score: float
    rerank_score: float


def tokenize(text: str) -> list[str]:
    """Normalize text into stable alphanumeric tokens shared by both representations."""
    return TOKEN_PATTERN.findall(text.lower())


def dense_vector(text: str, dimensions: int = DENSE_DIMENSIONS) -> list[float]:
    """Build a normalized signed hashing vector with no model and no external service."""
    values = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values[int.from_bytes(digest[:4], "big") % dimensions] += 1.0 if digest[4] % 2 else -1.0
    magnitude = math.sqrt(sum(value * value for value in values))
    return values if magnitude == 0 else [value / magnitude for value in values]


def sparse_vector(text: str) -> SparseVector:
    """Build a deterministic term-frequency sparse vector for Qdrant named sparse search."""
    counts: Counter[int] = Counter()
    for token, count in Counter(tokenize(text)).items():
        counts[int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % SPARSE_VOCABULARY_SIZE] += count
    indices = sorted(counts)
    return SparseVector(indices=indices, values=[float(counts[index]) for index in indices])


def document_text(ruling: Ruling) -> str:
    """The indexed text: the subject line plus the merchandise description, never the holding."""
    return f"{ruling.subject}\n{ruling.description}"


def ensure_collection(client: QdrantClient) -> None:
    """Create the hybrid rulings collection if it does not already exist."""
    if COLLECTION not in {collection.name for collection in client.get_collections().collections}:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=DENSE_DIMENSIONS, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )


def index(client: QdrantClient, rulings: Iterable[Ruling], batch_size: int = 512) -> int:
    """Upsert rulings into the hybrid collection, returning how many were written."""
    ensure_collection(client)
    batch: list[PointStruct] = []
    written = 0
    for ruling in rulings:
        text = document_text(ruling)
        batch.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"cross:{ruling.ruling_number}")),
                vector={"dense": dense_vector(text), "sparse": sparse_vector(text)},
                payload={
                    "ruling_number": ruling.ruling_number,
                    "hs6": ruling.hs6,
                    "hts_codes": ruling.hts_codes,
                    "subject": ruling.subject,
                    "date": ruling.date,
                    "description": ruling.description,
                },
            )
        )
        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION, points=batch, wait=True)
            written += len(batch)
            batch = []
    if batch:
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
        written += len(batch)
    return written


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """Boost hits sharing query terms with the ruling's subject, keeping fused score as tie-break."""
    query_terms = set(tokenize(query))
    scored = [
        hit.model_copy(update={"rerank_score": hit.score + (len(query_terms & set(tokenize(hit.subject))) / len(query_terms) if query_terms else 0.0)})
        for hit in hits
    ]
    return sorted(scored, key=lambda hit: (hit.rerank_score, hit.score), reverse=True)


def search(query: str, top_k: int = 8, *, client: QdrantClient) -> list[Hit]:
    """Fuse dense and sparse candidates with RRF, rerank, and return the top `top_k` rulings.

    Both candidate pools have a floor independent of `top_k`, so asking for one ruling and asking
    for eight rank the same candidates rather than fusing a different-sized pool per caller.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    ensure_collection(client)
    prefetch_limit = max(top_k * 3, MINIMUM_PREFETCH)
    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vector(query), using="dense", limit=prefetch_limit),
            Prefetch(query=sparse_vector(query), using="sparse", limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=max(top_k, RERANK_CANDIDATES),
        with_payload=True,
    )
    hits = [Hit(**point.payload, score=float(point.score), rerank_score=float(point.score)) for point in response.points]
    return rerank(query, hits)[:top_k]
