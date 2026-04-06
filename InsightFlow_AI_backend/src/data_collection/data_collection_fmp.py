import requests, os

FMP_API_KEY = os.getenv("FMP_API_KEY")

def collect_fmp_data():
    cre_companies = ["CBRE", "JLL", "BXP", "SPG", "ARE"]
    results = []
    for ticker in cre_companies:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = r.json()[0]
            results.append({
                "source": "FMP",
                "title": f"{d['companyName']} company profile",
                "content": d.get("description", "")[:1500],
                "link": d.get("website"),
                "entities": [d["companyName"]],
                "location": d.get("city", "Unknown"),
            })
    return results