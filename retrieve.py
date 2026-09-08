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
# Function words carry no classification signal, but in a raw term-frequency vector they are the
# loudest thing in a long document: a ruling whose subject runs 69 tokens scores on "of", "and" and
# "for" against every query in the corpus, which is how a jigsaw puzzle retrieves a radio terminal.
# Single characters go with them - "a", the "s" left by an apostrophe, a bare digit out of "ages 3".
STOPWORDS = frozenset(
    "about above after all also an and any are as at be been before being below between both but by "
    "can could did do does doing during each few for from further had has have having he her here "
    "him his how if in into is it its me more most my no nor of off on once only or other our out "
    "over own same she should so some such than that the their them then there these they this those "
    "through to too under until up us very was we were what when where which while who whom why will "
    "with would you your".split()
)
MINIMUM_PREFETCH = 40
RERANK_CANDIDATES = 40
# Bumped whenever tokenizing or either encoder changes, because vectors written by an older version
# are not comparable with vectors built by this one. The seed stamp carries it, so a stale index
# rebuilds itself instead of silently answering with the old representation.
ENCODER_VERSION = 2


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
    found = TOKEN_PATTERN.findall(text.lower())
    kept = [token for token in found if len(token) > 1 and token not in STOPWORDS]
    # A query that is nothing but function words still has to retrieve something rather than send an
    # empty vector to Qdrant, so the stoplist yields rather than empties the query.
    return kept or found


def relevant(query: str, hits: list["Hit"]) -> list["Hit"]:
    """Drop hits that share no term with the query.

    Retrieval has no relevance floor: `search` returns its top_k whether or not anything is close,
    so a query the corpus knows nothing about still comes back with the least-bad five. Requiring
    one shared token is a floor the user can verify by eye - the word they typed is in the row -
    and it is cheap, because the fused score itself cannot distinguish a weak match from no match.
    """
    terms = tokenize(query)
    if not terms:
        return []
    # The last token is usually half-typed in the live panel, so it matches as a prefix: "jigs"
    # keeps the jigsaw rulings on screen instead of blanking the panel until the word is finished.
    exact, typing = set(terms), terms[-1]
    kept = []
    for hit in hits:
        tokens = set(tokenize(f"{hit.subject} {hit.description}"))
        if exact & tokens or any(token.startswith(typing) for token in tokens):
            kept.append(hit)
    return kept


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
        # Sublinear term frequency: the tenth mention of "cotton" says much less than the first, and
        # raw counts let one repetitive ruling outscore an exactly-on-point one.
        counts[int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % SPARSE_VOCABULARY_SIZE] += 1.0 + math.log(count)
    # L2 normalised so a long ruling does not outrank a short one on length alone. Without this the
    # dot product rewards saying more, which in a corpus of legal prose is not the same as matching.
    magnitude = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    indices = sorted(counts)
    return SparseVector(indices=indices, values=[counts[index] / magnitude for index in indices])


def document_text(ruling: Ruling) -> str:
    """The indexed text: the subject line plus the merchandise description, never the holding."""
    return f"{ruling.subject}\n{ruling.description}"


def point_id(ruling_number: str) -> str:
    """The point's id, derived from the ruling number so a ruling can be addressed without a scan."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cross:{ruling_number}"))


def get(ruling_number: str, *, client: QdrantClient) -> Hit | None:
    """Fetch one indexed ruling by number, or None when the corpus does not have it.

    The payload carries everything a ruling page renders, so serving that page needs neither the
    corpus file nor an in-memory copy of it.
    """
    ensure_collection(client)
    # CROSS numbers are not consistently cased: rulings before roughly 2000 are stored lowercase
    # ("g83576") and later ones uppercase ("N261740"). A URL, a citation or a link may carry either,
    # so all three spellings are fetched at once rather than trusting the caller's case.
    spellings = dict.fromkeys([ruling_number, ruling_number.upper(), ruling_number.lower()])
    points = client.retrieve(collection_name=COLLECTION, ids=[point_id(s) for s in spellings], with_payload=True)
    return Hit(**points[0].payload, score=0.0, rerank_score=0.0) if points else None


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
                id=point_id(ruling.ruling_number),
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
