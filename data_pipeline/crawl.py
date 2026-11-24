"""Polite BFS crawler: robots.txt aware, rate limited, denoise filters, same-prefix only.

Usage:
    python -m data_pipeline.crawl                 # crawl all sources
    python -m data_pipeline.crawl --only tum-cs   # crawl one source
    python -m data_pipeline.crawl --clean         # wipe source dirs before re-crawling
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
import yaml
from bs4 import BeautifulSoup

from settings import RAW_DIR, ROOT, USER_AGENT

SKIP_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".mp4", ".mp3",
    ".ics", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".css", ".js",
)
TRACKING_PARAMS = {"pk_campaign", "pk_kwd", "pk_source", "pk_medium", "utm_source", "utm_medium", "utm_campaign"}
GLOBAL_EXCLUDE_URL = (
    "datenschutz", "privacy", "impressum", "imprint", "disclaimer", "cookie",
    "barrierefrei", "accessibility", "login", "logout", "sitemap", "kontakt",
    "contact", "news", "presse", "press", "jobs", "career", "karriere",
)
GLOBAL_EXCLUDE_TITLE = (
    "datenschutz", "data protection", "privacy policy", "imprint", "impressum",
    "resetting your password", "blank page", "wrong data entered", "incorrectly entered",
    "wrong degree programme", "doctoral candidates", "teaching degree", "joint bachelor",
)


class RobotsCache:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.cache: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.cache:
            rp = RobotFileParser()
            try:
                r = self.client.get(base + "/robots.txt", timeout=10)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self.cache[base] = rp
        return self.cache[base].can_fetch(USER_AGENT, url)


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    qs = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING_PARAMS]
    path = p.path.rstrip("/") if p.path.endswith("/") and p.path != "/" else p.path
    return urlunparse((p.scheme, p.netloc, path, p.params, urlencode(qs), ""))


def guess_lang(url: str, html_lang: str | None) -> str:
    if html_lang:
        return html_lang.split("-")[0].lower()
    u = url.lower()
    if "/en/" in u or ".en." in u or "set_language=en" in u or "/english/" in u or "lidx=1" in u:
        return "en"
    if "/zh" in u or "/cn" in u:
        return "zh"
    return "de"


def extract_html(url: str, html: str) -> tuple[str, str, str | None, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    html_lang = soup.html.get("lang") if soup.html else None
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    text = trafilatura.extract(
        html, url=url, output_format="markdown", include_tables=True,
        include_links=False, favor_recall=True,
    ) or ""
    return title, text, html_lang, links


def extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def link_ok(url: str, src: dict) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    if low.endswith(SKIP_EXT):
        return False
    if low.endswith(".pdf") and not src.get("include_pdfs", False):
        return False
    if not any(url.startswith(p) for p in src["allow_prefixes"]):
        return False
    excludes = list(GLOBAL_EXCLUDE_URL) + list(src.get("exclude_keywords", []))
    if any(ex.lower() in low for ex in excludes):
        return False
    kws = src.get("link_keywords")
    return not kws or any(k.lower() in low for k in kws)


def title_noisy(title: str, src: dict) -> bool:
    low = title.lower()
    excludes = list(GLOBAL_EXCLUDE_TITLE) + list(src.get("exclude_titles", []))
    return any(ex.lower() in low for ex in excludes)


def crawl_source(src: dict, defaults: dict, client: httpx.Client, robots: RobotsCache, clean: bool = False) -> int:
    max_pages = src.get("max_pages", defaults["max_pages"])
    max_depth = src.get("max_depth", defaults["max_depth"])
    delay = src.get("delay_seconds", defaults["delay_seconds"])
    out_dir = RAW_DIR / src["id"]
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queue = deque((normalize(u), 0) for u in src["seeds"])
    seen = {u for u, _ in queue}
    saved = 0
    while queue and saved < max_pages:
        url, depth = queue.popleft()
        if not robots.allowed(url):
            print(f"  [robots] skip {url}")
            continue
        try:
            r = client.get(url, timeout=20, follow_redirects=True)
        except Exception as e:
            print(f"  [error] {url}: {e}")
            continue
        time.sleep(delay)
        if r.status_code != 200:
            print(f"  [{r.status_code}] {url}")
            continue

        ctype = r.headers.get("content-type", "")
        links: list[str] = []
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            title, text, html_lang = url.rsplit("/", 1)[-1], extract_pdf(r.content), None
        elif "html" in ctype:
            title, text, html_lang, links = extract_html(url, r.text)
        else:
            continue

        if depth > 0 and title_noisy(title, src):
            print(f"  [denoise] skip {title[:55]!r}")
        elif len(text.strip()) >= 200:
            doc = {
                "id": hashlib.sha1(url.encode()).hexdigest()[:16],
                "url": url,
                "title": title,
                "text": text,
                "lang": guess_lang(url, html_lang),
                "source_id": src["id"],
                "category": src["category"],
                "university": src.get("university", ""),
                "program": src.get("program", ""),
                "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            (out_dir / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            saved += 1
            print(f"  [ok {saved}/{max_pages}] {title[:60]!r} {url}")

        if depth < max_depth:
            for link in links:
                link = normalize(link)
                if link not in seen and link_ok(link, src):
                    seen.add(link)
                    queue.append((link, depth + 1))
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="source ids to crawl")
    ap.add_argument("--clean", action="store_true", help="wipe target source dir before crawling")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "data_pipeline" / "sources.yaml").read_text(encoding="utf-8"))
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en,de;q=0.8,zh;q=0.5"}
    with httpx.Client(headers=headers) as client:
        robots = RobotsCache(client)
        for src in cfg["sources"]:
            if args.only and src["id"] not in args.only:
                continue
            print(f"== {src['id']}")
            n = crawl_source(src, cfg["defaults"], client, robots, clean=args.clean)
            print(f"== {src['id']}: saved {n} pages")


if __name__ == "__main__":
    main()
