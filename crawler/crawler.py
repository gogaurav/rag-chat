import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# BASE_URL = "https://about.gitlab.com/handbook/"
BASE_URL = "https://handbook.gitlab.com/"
OUTPUT_DIR = "gitlab_handbook"

visited = set()

# Create folder to store downloaded pages
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_handbook_url(url):
    """Check if URL is inside the GitLab handbook."""
    parsed = urlparse(url)
    return parsed.netloc == "handbook.gitlab.com" and "/handbook/" in parsed.path

def download_page(url):
    """Download an HTML page and return BeautifulSoup object."""
    try:
        print(f"Fetching: {url}")
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def save_page(url, html):
    """Save HTML content to local file mirroring structure."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # Convert path to filename
    if path.endswith("/"):
        path += "index.html"
    elif not path.endswith(".html"):
        path += ".html"

    output_path = os.path.join(OUTPUT_DIR, path)

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

def crawl(url):
    """Recursively crawl all handbook pages."""
    if url in visited:
        return

    visited.add(url)
    html = download_page(url)
    if not html:
        return

    save_page(url, html)

    soup = BeautifulSoup(html, "html.parser")

    # Extract all internal links
    for link in soup.find_all("a", href=True):
        next_url = urljoin(url, link["href"])

        if is_handbook_url(next_url) and next_url not in visited:
            print("hello")
            crawl(next_url)


if __name__ == "__main__":
    print("Starting GitLab Handbook crawl...")
    crawl(BASE_URL)
    print("Done! All pages saved to:", OUTPUT_DIR)
