import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://www.altusgroup.com"
INSIGHTS_URL = "https://www.altusgroup.com/insights/"

headers = {
    "User-Agent": "Mozilla/5.0"
}


# 🔹 Step 1: Simple scraping (no aggressive filters)
def fetch_altus_links():
    links = set()

    try:
        res = requests.get(INSIGHTS_URL, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "/insights/" in href:
                full_link = href if href.startswith("http") else urljoin(BASE_URL, href)
                links.add(full_link)

    except Exception as e:
        print("Error:", e)

    print(f"🔍 Basic scraped links: {len(links)}")
    return list(links)


# 🔹 Step 2: Strong fallback (IMPORTANT)
def fallback_links():
    return [
        "https://www.altusgroup.com/insights/commercial-real-estate-data-integration/",
        "https://www.altusgroup.com/insights/top-3-data-hurdles-real-estate-valuations-process/",
        "https://www.altusgroup.com/insights/cre-market-trends/",
        "https://www.altusgroup.com/insights/valuation-risk-management/",
        "https://www.altusgroup.com/insights/investment-market-outlook/",
        "https://www.altusgroup.com/insights/real-estate-lending-analysis/",
        "https://www.altusgroup.com/insights/property-market-data-strategy/",
        "https://www.altusgroup.com/insights/global-cre-insights/",
        "https://www.altusgroup.com/insights/data-driven-real-estate-decisions/",
        "https://www.altusgroup.com/insights/portfolio-risk-analysis/"
    ]


# 🔹 Step 3: Scrape article
def scrape_article(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])

        if len(content) < 150:
            return None

        return {
            "title": title,
            "content": content[:1500],
            "link": url
        }

    except:
        return None


# 🔹 Step 4: Main pipeline
def collect_altus_data(limit=10):
    links = fetch_altus_links()

    # ALWAYS add fallback (important)
    links.extend(fallback_links())

    final_data = []
    seen_titles = set()

    for link in links:
        if len(final_data) >= limit:
            break

        article = scrape_article(link)

        if article and article["title"] not in seen_titles:
            seen_titles.add(article["title"])

            final_data.append({
                "source": "Altus",
                "title": article["title"],
                "content": article["content"],
                "link": article["link"]
            })

            print(f"✅ Collected: {article['title'][:60]}")

            time.sleep(1)

    return final_data


# 🔹 Run
if __name__ == "__main__":
    data = collect_altus_data(limit=10)

    print("\n🎯 FINAL OUTPUT\n")

    for d in data[:3]:
        print(d)
        print("-" * 80)

    print(f"\n✅ Total Altus Articles Collected: {len(data)}")