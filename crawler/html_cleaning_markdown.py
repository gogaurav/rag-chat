#!/usr/bin/env python3
"""
clean_and_md.py

Walks a directory of downloaded HTML pages (from the crawler),
cleans navigation/footer/sidebar/TOC, extracts main content, converts to Markdown,
and saves cleaned HTML, Markdown, and a small metadata JSON per page.

Usage:
    python clean_and_md.py --input_dir gitlab_handbook --output_dir gitlab_handbook_cleaned
"""

import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from tqdm import tqdm

# Try to import markdownify, fallback to html2text
try:
    from markdownify import markdownify as mdify
    HAVE_MARKDOWNIFY = True
except Exception:
    HAVE_MARKDOWNIFY = False
    import html2text

# --- Configuration ---
INPUT_DIR = "gitlab_handbook"
OUTPUT_DIR = "gitlab_handbook_cleaned"
METADATA_FILE = "metadata.json"

# CSS selectors and attributes to remove as noise
NOISE_SELECTORS = [
    "nav", "header", "footer", "aside",
    ".sidebar", ".site-sidebar", ".toc", ".table-of-contents",
    ".breadcrumb", ".breadcrumbs", ".site-header", ".site-footer",
    ".nav", ".navigation", ".site-navigation", ".toc-container",
    ".promo", ".related", ".secondary", ".share", ".social",
    ".cookie", ".cookie-banner", ".newsletter", ".page-actions",
    "[role='navigation']",
    "[aria-label='breadcrumb']",
    "[aria-label='Table of contents']",
]

# Attributes to strip from remaining tags
ATTRS_TO_REMOVE = ["class", "id", "style", "aria-hidden", "data-*", "role"]


# --- Helpers ---
def remove_noisy_elements(soup):
    """Remove elements matched by NOISE_SELECTORS."""
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    # Remove script and style tags
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    return soup


def strip_attributes(soup):
    """Strip non-essential attributes to reduce noise."""
    for tag in soup.find_all(True):
        # Collect attrs to delete
        to_del = []
        for a in list(tag.attrs.keys()):
            if a in ("href", "src", "alt", "title"):  # keep important ones
                continue
            # wildcard match for data-* attributes
            if a.startswith("data-") or a in ATTRS_TO_REMOVE:
                to_del.append(a)
        for a in to_del:
            del tag.attrs[a]
    return soup


def get_main_content(soup):
    """Find main/article content using multiple heuristics."""
    # 1. <main>
    main = soup.find("main")
    if main and get_visible_text_length(main) > 100:
        return main

    # 2. <article>
    article = soup.find("article")
    if article and get_visible_text_length(article) > 100:
        return article

    # 3. role="main"
    role_main = soup.find(attrs={"role": "main"})
    if role_main and get_visible_text_length(role_main) > 100:
        return role_main

    # 4. Largest <div> by text length
    divs = soup.find_all("div")
    if divs:
        largest = max(divs, key=get_visible_text_length)
        if get_visible_text_length(largest) > 200:
            return largest

    # 5. Fallback to body
    return soup.body or soup


def get_visible_text_length(tag):
    """Return length of visible text (approx)."""
    if tag is None:
        return 0
    text = tag.get_text(separator=" ", strip=True)
    return len(text)


def clean_html(html):
    """Return cleaned HTML string and BeautifulSoup object of main content."""
    soup = BeautifulSoup(html, "html.parser")
    soup = remove_noisy_elements(soup)
    soup = strip_attributes(soup)
    main = get_main_content(soup)

    # Wrap the main into a minimal clean soup for saving
    clean_soup = BeautifulSoup("<!doctype html><html><head><meta charset='utf-8'></head><body></body></html>", "html.parser")
    if main:
        clean_soup.body.append(main)
    return str(clean_soup), main


def html_to_markdown(html_fragment):
    """Convert HTML fragment (string or Tag) to Markdown."""
    if not html_fragment:
        return ""
    if hasattr(html_fragment, "prettify"):
        html_str = str(html_fragment)
    else:
        html_str = html_fragment

    if HAVE_MARKDOWNIFY:
        # markdownify preserves some structure better
        md = mdify(html_str, heading_style="ATX")
    else:
        # fallback: html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        md = h.handle(html_str)
    # Basic cleanup: collapse multiple blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def normalize_path(input_path: Path, input_root: Path, output_root: Path):
    """Map input file path to output relative path preserving structure."""
    rel = input_path.relative_to(input_root)
    out_html_path = output_root / rel.with_suffix(".clean.html")
    out_md_path = output_root / rel.with_suffix(".md")
    out_json_path = output_root / rel.with_suffix(".json")
    return out_html_path, out_md_path, out_json_path


# --- Main processing ---
def process_file(in_path: Path, input_root: Path, output_root: Path):
    """Process a single HTML file: clean, convert, save metadata."""
    try:
        with in_path.open("r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        print(f"Failed to read {in_path}: {e}")
        return None

    cleaned_html, main_tag = clean_html(html)
    markdown = html_to_markdown(main_tag)

    # Extract title (if available)
    title = None
    try:
        soup_orig = BeautifulSoup(html, "html.parser")
        if soup_orig.title and soup_orig.title.string:
            title = soup_orig.title.string.strip()
        else:
            # try h1 in main
            if main_tag:
                h1 = main_tag.find("h1")
                if h1:
                    title = h1.get_text(strip=True)
    except Exception:
        title = None

    out_html_path, out_md_path, out_json_path = normalize_path(in_path, input_root, output_root)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned HTML
    with out_html_path.open("w", encoding="utf-8") as f:
        f.write(cleaned_html)

    # Save markdown
    with out_md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title or ''}\n\n" + markdown if title else markdown)

    # Save metadata
    meta = {
        "input_path": str(in_path),
        "clean_html_path": str(out_html_path),
        "markdown_path": str(out_md_path),
        "title": title,
        "text_length": len(markdown),
    }
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def process_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_meta = []
    html_files = list(input_root.rglob("*.html"))
    if not html_files:
        print("No HTML files found in", input_root)
        return

    for f in tqdm(html_files, desc="Processing HTML files"):
        meta = process_file(f, input_root, output_root)
        if meta:
            all_meta.append(meta)

    # Save aggregate metadata
    with (output_root / METADATA_FILE).open("w", encoding="utf-8") as mf:
        json.dump(all_meta, mf, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_meta)} files. Output in: {output_root}")


# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Clean HTML files and convert to Markdown")
    p.add_argument("--input_dir", default=INPUT_DIR, help="Directory containing downloaded HTML files")
    p.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to write cleaned HTML/Markdown")
    args = p.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    process_all(INPUT_DIR, OUTPUT_DIR)
