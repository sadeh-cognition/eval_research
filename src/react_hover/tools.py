"""Wikipedia search tools for the ReAct agent.

Default backend is the ColBERTv2 wiki17_abstracts index used in the official
DSPy agents tutorial. If that host is unreachable, falls back to the public
Wikipedia API (different corpus / may lower HoVer scores).
"""

from __future__ import annotations

from typing import Literal

import dspy
from loguru import logger

COLBERT_URL = "http://20.102.90.50:2017/wiki17_abstracts"

# Shared cache of title -> abstract text (tutorial pattern).
DOCS: dict[str, str] = {}

Backend = Literal["auto", "colbert", "wikipedia"]


def _wiki_module():
    import wikipedia

    wikipedia.set_lang("en")
    # Avoid MediaWiki default UA rejections on some networks.
    wikipedia.set_user_agent("dspy-react-eval/0.1 (research; local)")
    return wikipedia


class SearchBackend:
    """Thin wrapper around ColBERTv2 or Wikipedia search."""

    def __init__(self, backend: Backend = "auto"):
        self.backend = self._resolve(backend)
        self._colbert = None
        if self.backend == "colbert":
            self._colbert = dspy.ColBERTv2(url=COLBERT_URL)
        logger.info("tools.backend_ready backend={}", self.backend)

    @staticmethod
    def _resolve(backend: Backend) -> Backend:
        if backend != "auto":
            return backend
        try:
            import requests

            requests.get(COLBERT_URL, timeout=3)
            logger.info("tools.colbert_reachable url={}", COLBERT_URL)
            return "colbert"
        except Exception:
            logger.warning(
                "tools.colbert_unreachable url={} falling_back=wikipedia "
                "(HoVer gold titles target 2017 abstracts; scores may differ)",
                COLBERT_URL,
            )
            return "wikipedia"

    def search(self, query: str, k: int) -> list[str]:
        if self.backend == "colbert":
            assert self._colbert is not None
            results = self._colbert(query, k=k)
            texts = [x["text"] for x in results]
            for result in texts:
                if " | " in result:
                    title, text = result.split(" | ", 1)
                    DOCS[title] = text
            return texts

        return self._wikipedia_search(query, k)

    def lookup(self, title: str) -> str:
        if title in DOCS:
            return DOCS[title]

        if self.backend == "colbert":
            results = [x for x in self.search(title, 10) if x.startswith(title + " | ")]
            if not results:
                return f"No Wikipedia page found for title: {title}"
            return results[0]

        return self._wikipedia_lookup(title)

    def _fetch_page_text(self, title: str) -> str | None:
        wikipedia = _wiki_module()
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            text = (page.summary or page.content or "")[:500]
            DOCS[page.title] = text
            DOCS[title] = text
            return f"{page.title} | {text}"
        except wikipedia.exceptions.DisambiguationError as exc:
            options = "; ".join(exc.options[:12])
            msg = f"Disambiguation for {title!r}. Options include: {options}"
            DOCS[title] = msg
            return f"{title} | {msg}"
        except wikipedia.exceptions.PageError:
            return None
        except Exception as exc:
            return f"{title} | (page fetch failed: {exc})"

    def _wikipedia_search(self, query: str, k: int) -> list[str]:
        wikipedia = _wiki_module()
        try:
            titles = wikipedia.search(query, results=k)
        except Exception as exc:
            return [f"Search error: {exc}"]

        out: list[str] = []
        for title in titles:
            if title in DOCS:
                out.append(f"{title} | {DOCS[title]}")
                continue
            fetched = self._fetch_page_text(title)
            if fetched is None:
                out.append(f"{title} | (no page)")
            else:
                out.append(fetched)
        return out

    def _wikipedia_lookup(self, title: str) -> str:
        fetched = self._fetch_page_text(title)
        if fetched is not None:
            return fetched

        # Fall back to search-then-exact-prefix (ColBERT tutorial shape).
        results = [x for x in self.search(title, 10) if x.startswith(title + " | ")]
        if results:
            return results[0]
        return f"No Wikipedia page found for title: {title}"


_BACKEND: SearchBackend | None = None


def configure_backend(backend: Backend = "auto") -> SearchBackend:
    global _BACKEND
    _BACKEND = SearchBackend(backend)
    return _BACKEND


def get_backend() -> SearchBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = SearchBackend("auto")
    return _BACKEND


def search(query: str, k: int) -> list[str]:
    return get_backend().search(query, k)


def search_wikipedia(query: str) -> list[str]:
    """Returns top-5 results and then the titles of the top-5 to top-30 results."""
    top_k = search(query, 30)
    if not top_k:
        return ["No results."]

    titles = [f"`{x.split(' | ')[0]}`" for x in top_k[5:30] if " | " in x]
    head = top_k[:5]
    if titles:
        return head + [f"Other retrieved pages have titles: {', '.join(titles)}."]
    return head


def lookup_wikipedia(title: str) -> str:
    """Returns the text of the Wikipedia page, if it exists."""
    return get_backend().lookup(title)
