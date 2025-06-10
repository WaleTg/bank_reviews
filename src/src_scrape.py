
# src/scrape.py
from google_play_scraper import reviews, Sort
import pandas as pd
import pathlib, json, time

APPS = {
    "CBE":  "net.awl.cbe.mobile",
    "BOA":  "com.tandm.boa",          # update if package-id differs
    "Dashen": "com.ics.dashenbank"
}

def fetch_bank_reviews(pkg, limit=400):
    all_reviews = []
    token = None
    while len(all_reviews) < limit:
        batch, token = reviews(
            pkg,
            lang='en',
            country='et',
            sort=Sort.NEWEST,
            count=200,
            continuation_token=token
        )
        all_reviews.extend(batch)
        if token is None: break
        time.sleep(0.5)  # polite crawl
    return all_reviews[:limit]

def main():
    out = pathlib.Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bank, pkg in APPS.items():
        r = fetch_bank_reviews(pkg, limit=450)
        for rec in r:
            rows.append(
                dict(
                    review = rec["content"],
                    rating = rec["score"],
                    date   = rec["at"].strftime("%Y-%m-%d"),
                    bank   = bank,
                    source = "GooglePlay"
                )
            )
        print(f"{bank}: {len(r)} reviews")
    pd.DataFrame(rows).to_csv("data/raw/reviews_raw.csv", index=False)

if __name__ == "__main__":
    main()

