import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://www.jll.com"
INSIGHTS_URL = "https://www.jll.com/en-uk/insights"


# 🔹 Step 1: Get ALL article links (JS-rendered)
def fetch_jll_links():
    links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(INSIGHTS_URL, timeout=60000)

        # Scroll to load more content
        for _ in range(5):
            page.mouse.wheel(0, 5000)
            time.sleep(2)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "/en-uk/insights/" in href:
                full_link = href if href.startswith("http") else urljoin(BASE_URL, href)
                links.add(full_link)

        browser.close()

    print(f"🔍 Total links found (Playwright): {len(links)}")
    return list(links)


# 🔹 Step 2: Scrape article content
def scrape_article(page, url):
    try:
        page.goto(url, timeout=60000)
        time.sleep(2)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Date
        date = ""
        time_tag = soup.find("time")
        if time_tag:
            date = time_tag.get_text(strip=True)

        # Content
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])

        if len(content) < 150:
            return None

        return {
            "title": title,
            "date": date,
            "content": content[:1500],
            "link": url
        }

    except Exception as e:
        print(f"❌ Error: {url}")
        return None


# 🔹 Step 3: Main pipeline
def collect_jll_data(limit=12):
    links = fetch_jll_links()

    final_data = []
    seen_titles = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for link in links:
            if len(final_data) >= limit:
                break

            article = scrape_article(page, link)

            if article and article["title"] not in seen_titles:
                seen_titles.add(article["title"])

                final_data.append({
                    "source": "JLL",
                    "title": article["title"],
                    "date": article["date"],
                    "content": article["content"],
                    "link": article["link"]
                })

                print(f"✅ Collected: {article['title'][:60]}")

        browser.close()

    return final_data


# 🔹 Run
if __name__ == "__main__":
    data = collect_jll_data(limit=10)

    print("\n🎯 FINAL OUTPUT\n")

    for d in data[:3]:
        print(d)
        print("-" * 80)

    print(f"\n✅ Total JLL Articles Collected: {len(data)}")