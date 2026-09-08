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


def test_salutation_on_a_shared_line_still_cuts_the_header():
    modern = RULING_TEXT.replace("\r\r", "  ").replace("\r", " ")
    description = cross.extract_description(modern)
    assert "cotton jersey knit" in description
    assert not cross.HTS_CODE.findall(description)
    assert len(description) > cross.MINIMUM_DESCRIPTION_CHARACTERS


def test_trade_remedy_provisions_are_not_classification_answers():
    assert cross.normalize_codes("6505.00.6090, 9903.01.24, 9903.88.03") == ["6505006090"]
    assert cross.normalize_codes("9903.01.25") == []


def test_ruling_page_renders_fully_linked_and_escaped():
    import app

    rulings = [
        make_ruling("N1", "6109100012", "cotton t-shirt <script>alert(1)</script>", "a men's t-shirt of 100% cotton jersey knit fabric, short sleeves, crew neck"),
        make_ruling("N2", "6110202079", "knit cotton pullover", "a men's pullover sweater of 100% cotton knit fabric with long sleeves"),
        make_ruling("N3", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated, in retail bags"),
    ]
    client = QdrantClient(":memory:")
    retrieve.index(client, rulings)
    app.state["client"] = client

    page = app.render_ruling(rulings[0])
    assert "{{" not in page, "an unreplaced template placeholder shipped to the crawler"
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert '<link rel="canonical" href="http://localhost:8099/ruling/N1"' in page
    assert '/ruling/N2' in page, "related rulings are what stop these pages being orphans"
    assert page.count('href="/ruling/N1"') == 0, "a ruling page must not list itself as related"

    app.state.clear()


def test_home_page_renders_both_verdicts_with_no_placeholders_left():
    import app

    app.state["indexed"] = 23929
    answered = classify.gate(make_result())
    withheld = classify.gate(make_result(confidence=0.42))

    for result, verdict in ((answered, "answered"), (withheld, "escalated")):
        page = app.render_page("cotton t-shirt", app.render_result(result))
        assert "{{" not in page, "an unreplaced placeholder shipped to the browser"
        assert f'class="verdict {verdict}"' in page
        assert "23,929" in page, "the corpus size is quoted on the page and must be the real one"
        assert '<link rel="canonical" href="http://localhost:8099/"' in page, "canonical must drop the query"
        assert 'href="/ruling/N1"' in page

    # No link on the page may carry a query: /?q= is a GET that costs a model call, and a
    # prefetcher, a crawler or a refresh would fire it with nobody having pressed Classify.
    page = app.render_page("", "")
    assert 'href="/?q=' not in page
    assert page.count('class="chip" data-q=') == 4

    assert '<p class="code">610910</p>' in app.render_page("x", app.render_result(answered))
    assert 'class="code"' not in app.render_page("x", app.render_result(withheld)), "a withheld answer must never render the code"
    app.state.clear()


def test_rate_limit_backoff_grows_and_is_capped():
    assert classify.retry_delay(0) == classify.BACKOFF_BASE_SECONDS
    assert classify.retry_delay(3) == classify.BACKOFF_BASE_SECONDS * 8
    assert classify.retry_delay(20) == classify.MAXIMUM_BACKOFF_SECONDS


def test_fallback_chain_moves_on_and_stamps_who_answered():
    import httpx

    class Dead:
        model = "dead-model"

        def complete_json(self, *, system, user):
            raise httpx.HTTPError("rate limited")

    class Alive:
        model = "live-model"

        def __init__(self):
            self.calls = 0

        def complete_json(self, *, system, user):
            self.calls += 1
            return {"hs6": "610910", "reasoning": "r", "citations": ["N1"], "confidence": 0.95}

    alive = Alive()
    chain = classify.FallbackChatClient([Dead(), alive])
    assert chain.complete_json(system="s", user="u")["_served_by"] == "live-model"
    assert alive.calls == 1
    assert chain.model == "dead-model -> live-model"

    import pytest

    with pytest.raises(httpx.HTTPError):
        classify.FallbackChatClient([Dead(), Dead()]).complete_json(system="s", user="u")


def test_suggest_is_retrieval_only_and_ignores_a_fragment():
    import app

    rulings = [
        make_ruling("N1", "6109100012", "cotton t-shirt", "a men's t-shirt of 100% cotton jersey knit fabric, short sleeves, crew neck"),
        make_ruling("N2", "8471300100", "portable computer", "a portable laptop computer with an attached keyboard and a 13 inch screen"),
    ]
    client = QdrantClient(":memory:")
    retrieve.index(client, rulings)
    app.state["client"] = client

    assert app.suggest("co") == [], "below the floor nothing is retrieved, so nothing is shown"
    hits = app.suggest("cotton knit t-shirt")
    assert hits[0]["ruling_number"] == "N1" and hits[0]["hs6"] == "610910"
    assert set(hits[0]) == {"ruling_number", "hs6", "subject"}, "suggestions must not leak a holding"
    app.state.clear()


def test_seed_rebuilds_only_when_the_corpus_file_changes(tmp_path, monkeypatch):
    import app

    coffee = make_ruling("N1", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated, in retail bags")
    shirt = make_ruling("N2", "6109100012", "cotton t-shirt", "a men's t-shirt of 100% cotton jersey knit fabric, short sleeves, crew neck")
    corpus = tmp_path / "rulings.jsonl"
    corpus.write_text(coffee.model_dump_json() + "\n")
    monkeypatch.setattr(app, "RULINGS_FILE", corpus)
    monkeypatch.setattr(app, "SEED_STAMP", tmp_path / "qdrant.seed")
    monkeypatch.setattr(app, "SITEMAP_FILE", tmp_path / "qdrant.sitemap.xml")
    monkeypatch.setattr(app, "QDRANT_PATH", tmp_path / "qdrant")

    client = QdrantClient(":memory:")
    assert app.seed(client) == 1, "an unseeded index must be built"

    # An unchanged corpus must not be re-read at all: emptying the file would be visible if it were.
    stamp = app.corpus_stamp()
    corpus.write_text("")
    import os
    os.utime(corpus, ns=(0, int(stamp.split(":")[1])))
    monkeypatch.setattr(app, "corpus_stamp", lambda: stamp)
    assert app.seed(client) == 1, "an unchanged corpus must reuse the stored index rather than rebuild"

    monkeypatch.undo()
    monkeypatch.setattr(app, "RULINGS_FILE", corpus)
    monkeypatch.setattr(app, "SEED_STAMP", tmp_path / "qdrant.seed")
    monkeypatch.setattr(app, "SITEMAP_FILE", tmp_path / "qdrant.sitemap.xml")
    monkeypatch.setattr(app, "QDRANT_PATH", tmp_path / "qdrant")
    corpus.write_text(coffee.model_dump_json() + "\n" + shirt.model_dump_json() + "\n")
    assert app.seed(client) == 2, "a changed corpus must rebuild"
    assert "/sitemap-0.xml</loc>" in (tmp_path / "qdrant.sitemap.xml").read_text(), "the index names its parts"
    assert "/ruling/N2</loc>" in app.sitemap_part(0).read_text(), "the URLs live in the parts"


def test_a_ruling_page_is_served_from_the_index_not_the_corpus_file():
    """The corpus file is nearly a gigabyte; a ruling page must never need it."""
    import app

    rulings = [
        make_ruling("N1", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated, in retail bags"),
        make_ruling("N2", "6109100012", "cotton t-shirt", "a men's t-shirt of 100% cotton jersey knit fabric, short sleeves, crew neck"),
    ]
    client = QdrantClient(":memory:")
    retrieve.index(client, rulings)

    fetched = retrieve.get("N1", client=client)
    assert fetched is not None and fetched.subject == "roasted coffee" and fetched.hs6 == "090121"
    assert retrieve.get("NOPE", client=client) is None, "an unknown ruling number must 404, not raise"

    app.state["client"] = client
    page = app.render_ruling(fetched)
    assert "{{" not in page and "roasted arabica" in page
    assert '/ruling/N2' in page, "related rulings still come from retrieval"
    app.state.clear()


def test_a_brand_query_is_answered_with_the_spread_not_an_instruction():
    import app

    rulings = [
        make_ruling("N1", "9503000000", "The tariff classification of LEGO minifigure toys", "Moulded plastic minifigure toys representing characters, put up for retail sale."),
        make_ruling("N2", "6104630000", "The tariff classification of a LEGO costume from China", "A child's costume of knit polyester consisting of a pullover and trousers resembling a character."),
        make_ruling("N3", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated, in retail bags"),
    ]
    client = QdrantClient(":memory:")
    retrieve.index(client, rulings)
    app.state["client"] = client
    app.state["indexed"] = 3

    page = app.render_page("Lego", app.render_too_short("Lego"))
    assert "{{" not in page
    assert 'href="/ruling/N1"' in page and 'href="/ruling/N2"' in page, "the rulings behind the refusal must be shown"
    assert "2 different chapters" in page, "the spread across chapters is the reason more detail is needed"
    assert 'class="code"' not in page, "a query too short to classify must never render a code"

    # Too short even to retrieve on: guidance only, and no empty table.
    assert "<table" not in app.render_too_short("ab")
    app.state.clear()


def test_sitemap_splits_at_the_cap_and_the_index_names_every_part(tmp_path, monkeypatch):
    """A sitemap over 50,000 URLs is rejected whole, so the corpus must be chunked behind an index."""
    import app

    monkeypatch.setattr(app, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(app, "SITEMAP_FILE", tmp_path / "qdrant.sitemap.xml")
    monkeypatch.setattr(app, "SITEMAP_URLS_PER_FILE", 3)

    rulings = [make_ruling(f"N{n}", "0901210000", f"ruling {n}", "roasted arabica coffee beans in retail bags") for n in range(7)]
    # 7 rulings plus the home page URL, three to a part.
    assert app.build_sitemap(rulings) == 3

    index = (tmp_path / "qdrant.sitemap.xml").read_text()
    assert index.count("<sitemap>") == 3 and "<urlset" not in index
    assert all(f"/sitemap-{n}.xml</loc>" in index for n in range(3))

    parts = "".join(app.sitemap_part(n).read_text() for n in range(3))
    assert all(f"/ruling/N{n}</loc>" in parts for n in range(7)), "every ruling must appear in some part"
    assert parts.count("<url>") == 8, "the home page plus every ruling, once each"
    assert app.sitemap_part(0).read_text().count("<url>") == 3, "parts must respect the cap"

    # A corpus that shrinks must not leave parts the index no longer names.
    assert app.build_sitemap(rulings[:2]) == 1
    assert not app.sitemap_part(1).exists() and not app.sitemap_part(2).exists()


def test_a_ruling_page_resolves_whatever_case_the_number_is_written_in():
    """CROSS numbers are lowercase before ~2000 and uppercase after; 21.6% of the corpus is lowercase."""
    import app

    rulings = [
        make_ruling("g83576", "6109100012", "cotton t-shirt", "a men's t-shirt of 100% cotton jersey knit fabric, short sleeves"),
        make_ruling("N261740", "0901210000", "roasted coffee", "roasted arabica coffee beans, not decaffeinated, in retail bags"),
    ]
    client = QdrantClient(":memory:")
    retrieve.index(client, rulings)

    for asked in ("g83576", "G83576", "N261740", "n261740"):
        assert retrieve.get(asked, client=client) is not None, f"/ruling/{asked} must resolve, not 404"
    assert retrieve.get("NOPE123", client=client) is None, "an unknown number must still 404"


def test_a_malformed_subheading_never_reaches_the_result():
    """The gate withholds a bad code, but the bad string must not survive on the object either."""
    import classify

    valid = classify.Classification(hs6="090121", reasoning="", citations=[], confidence=0.5,
                                    disposition="escalated", reason="", candidates=[])
    assert valid.hs6 == "090121"

    for junk in ("UNKNOW", "unknown", "N/A", "chapter 09", "12345", "", None):
        result = classify.Classification(hs6=junk, reasoning="", citations=[], confidence=0.5,
                                         disposition="escalated", reason="", candidates=[])
        assert result.hs6 is None, f"{junk!r} is not a subheading and must not be carried"
        assert '"hs6":null' in result.model_dump_json(), "the API response must not leak it either"

    # Separators are normalised, not rejected: this is how the model usually writes a code.
    for written, expected in (("0901.21.0000", "090121"), ("6109.10", "610910"), ("0901 21 0000", "090121")):
        assert classify.Classification(hs6=written, reasoning="", citations=[], confidence=0.5,
                                       disposition="escalated", reason="", candidates=[]).hs6 == expected


def test_every_result_carries_the_ai_disclaimer():
    """A code must never be shown without it, on either verdict."""
    import app

    app.state["indexed"] = 176940
    for result in (classify.gate(make_result()), classify.gate(make_result(confidence=0.42))):
        page = app.render_page("cotton t-shirt", app.render_result(result))
        assert 'class="disclaimer"' in page
        assert "can be wrong" in page and "not customs advice" in page
        assert "licensed customs broker" in page
    app.state.clear()
