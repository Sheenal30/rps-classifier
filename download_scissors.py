# download_scissors_wikimedia.py
# Robust downloader for scissors / hand-scissors images from Wikimedia Commons.
# Run locally. Writes images to data/real/scissors and creates attributions.txt.
#
# Notes:
# - Uses the Commons API to find file pages and extracts direct image URLs.
# - Skips non-image results, 403/404, or very small files.
# - Converts images to JPG for consistency.

import pathlib, requests, time, sys
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, unquote

OUT_DIR = pathlib.Path("data/real/scissors")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ATTR_PATH = OUT_DIR / "attributions.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.0.0 Safari/537.36",
    "Referer": "https://commons.wikimedia.org/",
}

API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

# Config
QUERY = "hand scissors OR scissors gesture OR scissors hand"
TARGET = 30           # how many images you want
GSR_LIMIT = 50        # search page size per API call (max 50 for anonymous)
SLEEP = 0.2

def sanitize_filename(url):
    name = unquote(pathlib.Path(urlparse(url).path).name)
    # replace spaces and weird chars
    name = name.replace(" ", "_").replace("(", "").replace(")", "")
    if not (name.lower().endswith(".jpg") or name.lower().endswith(".jpeg")):
        # give it jpg extension
        name = name + ".jpg"
    return name

def get_image_pages(query, limit):
    """
    Use generator=search to fetch pages matching query and return list of page dicts.
    """
    pages = []
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": min(limit, GSR_LIMIT),
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
    }
    r = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "query" not in data:
        return []
    for pid, p in data["query"]["pages"].items():
        # Ensure there's imageinfo
        if "imageinfo" in p:
            pages.append(p)
    return pages

def download_image(url, outpath, max_retries=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
            r.raise_for_status()
            content = r.content
            img = Image.open(BytesIO(content)).convert("RGB")
            # skip tiny images
            w, h = img.size
            if w < 50 or h < 50:
                return False, "Too small"
            img.save(outpath, format="JPEG", quality=90)
            return True, None
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    return False, last_err

def main():
    print("Searching Wikimedia Commons for images...")
    pages = get_image_pages(QUERY, TARGET)
    if not pages:
        print("No pages found. Exiting.")
        return

    attributions = []
    saved = 0
    seen_urls = set()

    for p in pages:
        if saved >= TARGET:
            break
        imageinfo = p.get("imageinfo", [])
        if not imageinfo:
            continue
        ii = imageinfo[0]
        url = ii.get("url")
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        fname = sanitize_filename(url)
        outpath = OUT_DIR / fname
        print(f"Downloading {url} -> {fname}")
        ok, err = download_image(url, outpath)
        if ok:
            saved += 1
            # Try to capture license/title if present
            title = p.get("title", "")
            license_name = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            attributions.append(f"{fname} <- {url}   title: {title}   license: {license_name}")
            print("  saved")
        else:
            attributions.append(f"FAILED {fname} <- {url}   ERROR: {err}")
            print("  failed:", err)
        time.sleep(SLEEP)

    with open(ATTR_PATH, "w", encoding="utf-8") as f:
        f.write("Attributions and sources\n")
        f.write("\n".join(attributions))

    print(f"Saved {saved} images to {OUT_DIR}")
    print(f"Attributions written to {ATTR_PATH}")
    if saved == 0:
        print("No images saved. Try running from a different network or increase GSR_LIMIT / TARGET.")
        print("You can also request the augmentation-only route instead.")

if __name__ == "__main__":
    main()