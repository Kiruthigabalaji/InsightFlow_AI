# import requests
# from bs4 import BeautifulSoup

# url = "https://www.propertyweek.com/news"

# headers = {
#     "User-Agent": "Mozilla/5.0"
# }

# res = requests.get(url, headers=headers)
# soup = BeautifulSoup(res.text, "html.parser")

# articles = soup.find_all("a")

# data = []

# for a in articles:
#     title = a.get_text(strip=True)
#     link = a.get("href")

#     if title and link and "/news/" in link:
#         data.append({
#             "source": "PropertyWeek",
#             "title": title,
#             "link": link
#         })

# print(data[:10])
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.propertyweek.com"

headers = {
    "User-Agent": "Mozilla/5.0"
}


# 🔹 Step 1: Get article links
def fetch_article_links():
    url = f"{BASE_URL}/news"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    links = set()  # avoid duplicates

    for a in soup.find_all("a"):
        title = a.get_text(strip=True)
        href = a.get("href")

        if title and href and "/news/" in href:
            full_link = urljoin(BASE_URL, href)
            links.add((title, full_link))

    return list(links)


# 🔹 Step 2: Scrape article details
def scrape_article(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

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

        # Filter bad content
        if len(content) < 100:
            return None

        return {
            "title": title,
            "date": date,
            "content": content[:1500],  # limit size
            "link": url
        }

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None


# 🔹 Step 3: Combine everything
def collect_propertyweek_data(limit=10):
    articles = fetch_article_links()

    print(f"🔍 Found {len(articles)} raw links")

    final_data = []

    for title, link in articles[:limit]:
        article_data = scrape_article(link)

        if article_data:
            final_data.append({
                "source": "PropertyWeek",
                "title": article_data["title"] or title,
                "date": article_data["date"],
                "content": article_data["content"],
                "link": link
            })

    return final_data


# 🔹 Run
if __name__ == "__main__":
    data = collect_propertyweek_data(limit=10)

    print(f"\n✅ Final collected articles: {len(data)}\n")

    for d in data[:3]:
        print(d)
        print("-" * 80)