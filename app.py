"""The free single-lookup page and the JSON endpoint behind it.

The page is server-rendered rather than a client-side app for one reason: a result that only exists
after JavaScript runs is not a page a crawler indexes or a broker can paste into an email. Every
lookup has a shareable `/?q=` URL that renders its own answer.
"""
import html
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from classify import ANSWER_CONFIDENCE_THRESHOLD, Classification, classify, default_chat_client
from cross import load_rulings
from retrieve import index, search

STATIC = Path(__file__).resolve().parent / "static"
TEMPLATE = STATIC / "index.html"
RULING_TEMPLATE = STATIC / "ruling.html"
CROSS_URL = "https://rulings.cbp.gov/ruling/"
# Absolute URLs are required in a sitemap and in a canonical tag, and getting them wrong is how a
# site tells Google that 23,929 pages are really one page on localhost.
BASE_URL = os.getenv("BASE_URL", "http://localhost:8099").rstrip("/")
RELATED_RULINGS = 6
MINIMUM_QUERY_CHARACTERS = 12
MAXIMUM_QUERY_CHARACTERS = 4000

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the retrieval index once at startup, so a lookup is never charged for indexing."""
    # ponytail: in-memory Qdrant, which warns above 20k points and rebuilds on every restart.
    # Move to a Qdrant server when startup time or a second web process makes it hurt.
    client = QdrantClient(":memory:")
    rulings = load_rulings()
    state["indexed"] = index(client, rulings)
    state["client"] = client
    state["chat"] = default_chat_client()
    state["by_number"] = {ruling.ruling_number: ruling for ruling in rulings}
    state["sitemap"] = build_sitemap(rulings)
    yield
    state.clear()


app = FastAPI(title="HS classification", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class LookupRequest(BaseModel):
    """A merchandise description to classify."""

    description: str = Field(min_length=MINIMUM_QUERY_CHARACTERS, max_length=MAXIMUM_QUERY_CHARACTERS)


def lookup(description: str) -> Classification:
    """Classify one description against the loaded corpus."""
    return classify(description.strip(), client=state["client"], chat=state["chat"])


def render_result(result: Classification) -> str:
    """Render the gate's verdict as the headline, because the verdict is what is being sold.

    An escalation is presented as a result rather than as a failure: the rulings that were consulted
    are still shown, so a withheld answer still leaves the user better off than they started.
    """
    if result.disposition == "answered":
        headline = f"""<p class="verdict answered">Classified</p>
        <p class="code">{html.escape(result.hs6 or "")}</p>
        <p class="note">{html.escape(result.reason)}</p>"""
    else:
        headline = f"""<p class="verdict escalated">Not answered</p>
        <p class="note">{html.escape(result.reason)} A classification below the
        {ANSWER_CONFIDENCE_THRESHOLD:.0%} threshold is withheld rather than guessed, because a wrong
        HS code on an entry is a false statement to CBP under 19&nbsp;U.S.C.&nbsp;&sect;1592. The
        closest rulings are below; a licensed customs broker should make the call.</p>"""
    cited = set(result.citations)
    rulings = "".join(
        f"""<li class="{'cited' if hit.ruling_number in cited else ''}">
          <a href="/ruling/{html.escape(hit.ruling_number)}">{html.escape(hit.ruling_number)}</a>
          <span class="hs">{html.escape(hit.hs6)}</span>
          <span class="subject">{html.escape(hit.subject)}</span>
          <span class="date">{html.escape(hit.date)}</span>
        </li>"""
        for hit in result.candidates
    )
    return f"""<section class="result">
      {headline}
      <p class="confidence">Model confidence {result.confidence:.0%}</p>
      <h2>Reasoning</h2>
      <p class="reasoning">{html.escape(result.reasoning)}</p>
      <h2>CBP rulings consulted <span class="hint">cited rulings highlighted</span></h2>
      <ol class="rulings">{rulings}</ol>
    </section>"""


def render_page(query: str, body: str) -> str:
    """Substitute the query and rendered result into the static template.

    The canonical URL deliberately omits `q`: every distinct lookup would otherwise be a separate
    indexable page saying roughly the same thing, which is how a useful tool turns into thin content.
    """
    return (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("{{CANONICAL}}", f"{BASE_URL}/")
        .replace("{{QUERY}}", html.escape(query, quote=True))
        .replace("<!--RESULT-->", body)
    )


def build_sitemap(rulings: list) -> str:
    """Render the corpus as one sitemap, which is what makes these pages discoverable at all."""
    urls = "".join(
        f"<url><loc>{BASE_URL}/ruling/{quote(ruling.ruling_number)}</loc><lastmod>{ruling.date}</lastmod></url>"
        for ruling in rulings
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{BASE_URL}/</loc></url>{urls}</urlset>'


def render_ruling(ruling) -> str:
    """Render one ruling as an indexable page, linked to the rulings nearest it in the corpus.

    The related list is the point as much as the ruling text is: 23,929 orphan pages are 23,929
    pages nothing links to, and retrieval already knows which ones belong next to each other.
    """
    related = [hit for hit in search(ruling.description, top_k=RELATED_RULINGS + 1, client=state["client"]) if hit.ruling_number != ruling.ruling_number]
    codes = "".join(
        f'<li><span class="hs6">{html.escape(code[:6])}</span>{html.escape(code[6:])}</li>' for code in ruling.hts_codes
    )
    related_items = "".join(
        f'<li><a href="/ruling/{html.escape(hit.ruling_number)}">{html.escape(hit.subject)}</a><span class="hs">{html.escape(hit.hs6)}</span></li>'
        for hit in related[:RELATED_RULINGS]
    )
    summary = " ".join(ruling.description.split())[:155]
    return (
        RULING_TEMPLATE.read_text(encoding="utf-8")
        .replace("{{TITLE}}", html.escape(f"{ruling.subject} — CBP ruling {ruling.ruling_number} (HS {ruling.hs6})"))
        .replace("{{META}}", html.escape(f"HS {ruling.hs6}. {summary}", quote=True))
        .replace("{{CANONICAL}}", f"{BASE_URL}/ruling/{quote(ruling.ruling_number)}")
        .replace("{{SUBJECT}}", html.escape(ruling.subject))
        .replace("{{CODES}}", codes)
        .replace("{{DATE}}", html.escape(ruling.date))
        .replace("{{DESCRIPTION}}", html.escape(ruling.description))
        .replace("{{RELATED}}", related_items or "<li>No related rulings were found.</li>")
        .replace("{{QUERY}}", quote(ruling.subject))
        .replace("{{NUMBER}}", html.escape(ruling.ruling_number))
    )


@app.get("/", response_class=HTMLResponse)
def home(q: str = Query(default="", max_length=MAXIMUM_QUERY_CHARACTERS)) -> HTMLResponse:
    """Render the lookup form, and the result for `q` when one was asked for."""
    query = q.strip()
    if not query:
        return HTMLResponse(render_page("", ""))
    if len(query) < MINIMUM_QUERY_CHARACTERS:
        return HTMLResponse(render_page(query, '<section class="result"><p class="note">Describe the goods in a sentence or two — what it is, what it is made of, and what it is used for.</p></section>'))
    return HTMLResponse(render_page(query, render_result(lookup(query))))


@app.post("/api/classify")
def api_classify(request: LookupRequest = Body(...)) -> Classification:
    """Classify one description and return the full result, gate verdict included."""
    return lookup(request.description)


@app.get("/api/health")
def health() -> dict:
    """Report whether the corpus is loaded and whether a model is configured."""
    return {"indexed_rulings": state.get("indexed", 0), "model": bool(state.get("chat")), "threshold": ANSWER_CONFIDENCE_THRESHOLD}


@app.get("/ruling/{number}", response_class=HTMLResponse)
def ruling_page(number: str) -> HTMLResponse:
    """Render one CBP ruling as its own indexable page."""
    ruling = state["by_number"].get(number.upper())
    if ruling is None:
        raise HTTPException(status_code=404, detail="No such ruling in the corpus.")
    return HTMLResponse(render_ruling(ruling))


@app.get("/sitemap.xml")
def sitemap() -> Response:
    """Serve the corpus sitemap, built once at startup."""
    return Response(content=state["sitemap"], media_type="application/xml")


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    """Allow the corpus to be crawled, and keep lookup results out of the index."""
    return f"User-agent: *\nAllow: /\nDisallow: /?q=\nDisallow: /api/\nSitemap: {BASE_URL}/sitemap.xml\n"
