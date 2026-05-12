"""
Hespress.com Comment Scraper
============================
Scrapes Arabic/Darija comments from Hespress.com articles
for building a sentiment analysis dataset.

Author: Khalid Morjan
Project: Darija Sentiment Analysis
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import os
from datetime import datetime
from tqdm import tqdm


# ─── CONFIG ───────────────────────────────────────────────────────────────────

BASE_URL = "https://www.hespress.com"

# Categories to scrape (covers diverse topics = diverse sentiment)
CATEGORIES = [
    "/politique",
    "/societe", 
    "/economie",
    "/sport",
    "/faits-divers",
    "/regional",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.hespress.com/",
}

# Delay between requests (be respectful to the server)
MIN_DELAY = 1.5
MAX_DELAY = 3.5

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hespress_comments_raw.csv")
ARTICLES_FILE = os.path.join(OUTPUT_DIR, "articles_scraped.json")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_page(url, retries=3):
    """Fetch a page with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                print(f"  ⚠️  Rate limited — waiting 10s...")
                time.sleep(10)
            else:
                print(f"  ⚠️  Status {resp.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Attempt {attempt+1} failed: {e}")
        time.sleep(random.uniform(MIN_DELAY * 2, MAX_DELAY * 2))
    return None


def polite_sleep():
    """Random delay between requests to avoid being blocked."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# ─── SCRAPING FUNCTIONS ───────────────────────────────────────────────────────

def get_article_links(category_url, max_pages=5):
    """
    Extract article links from a category page.
    Hespress uses standard WordPress-style pagination.
    """
    links = []
    
    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = f"{BASE_URL}{category_url}"
        else:
            url = f"{BASE_URL}{category_url}/page/{page_num}"
        
        print(f"  📄 Fetching category page {page_num}: {url}")
        html = get_page(url)
        
        if not html:
            break
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Hespress article links are in <h2> or <h3> tags with class containing 'card-title'
        # or inside <article> tags
        article_cards = soup.find_all("article")
        
        if not article_cards:
            # Try alternative selectors
            article_cards = soup.find_all("div", class_=lambda c: c and "post" in c.lower())
        
        for card in article_cards:
            link_tag = card.find("a", href=True)
            if link_tag and "hespress.com" in link_tag["href"]:
                href = link_tag["href"]
                if href not in links and "/category/" not in href and "/tag/" not in href:
                    links.append(href)
        
        if not article_cards:
            print(f"  ℹ️  No articles found on page {page_num}, stopping.")
            break
            
        polite_sleep()
    
    print(f"  ✅ Found {len(links)} article links in {category_url}")
    return links


def scrape_article_comments(article_url):
    """
    Extract comments from a single Hespress article.
    Returns list of comment dicts.
    """
    html = get_page(article_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    comments = []
    
    # Get article title for context
    title_tag = soup.find("h1")
    article_title = title_tag.get_text(strip=True) if title_tag else ""
    
    # Get article category
    breadcrumb = soup.find("div", class_=lambda c: c and "breadcrumb" in str(c).lower())
    category = ""
    if breadcrumb:
        cat_links = breadcrumb.find_all("a")
        if len(cat_links) > 1:
            category = cat_links[-1].get_text(strip=True)
    
    # ── Comment selectors (Hespress uses WordPress comment system) ──
    # Try multiple selectors as the site may vary
    comment_containers = (
        soup.find_all("li", class_=lambda c: c and "comment" in str(c).lower()) or
        soup.find_all("div", class_=lambda c: c and "comment-body" in str(c).lower()) or
        soup.find_all("article", class_=lambda c: c and "comment" in str(c).lower())
    )
    
    for container in comment_containers:
        # Skip pingbacks/trackbacks
        if "pingback" in str(container.get("class", [])).lower():
            continue
        
        # Extract comment text
        text_tag = (
            container.find("div", class_=lambda c: c and "comment-content" in str(c).lower()) or
            container.find("p") or
            container.find("div", class_="text")
        )
        
        if not text_tag:
            continue
            
        text = text_tag.get_text(strip=True)
        
        # Filter: minimum 5 characters, skip empty
        if len(text) < 5:
            continue
        
        # Extract author name
        author_tag = container.find(class_=lambda c: c and "author" in str(c).lower())
        author = author_tag.get_text(strip=True) if author_tag else "anonymous"
        
        # Extract date
        date_tag = container.find("time") or container.find(class_=lambda c: c and "date" in str(c).lower())
        date = date_tag.get("datetime", date_tag.get_text(strip=True)) if date_tag else ""
        
        comments.append({
            "text": text,
            "author": author,
            "date": date,
            "article_title": article_title,
            "article_url": article_url,
            "category": category,
            "scraped_at": datetime.now().isoformat(),
            "label": ""  # To be filled during annotation
        })
    
    return comments


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_scraper(
    max_articles_per_category=30,
    max_pages_per_category=3
):
    """
    Full scraping pipeline:
    1. Collect article URLs from each category
    2. Scrape comments from each article
    3. Save to CSV
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_comments = []
    all_article_urls = []
    
    print("\n" + "="*60)
    print("🚀 Hespress Comment Scraper — Darija Sentiment Dataset")
    print("="*60 + "\n")
    
    # ── Step 1: Collect article URLs ──
    print("📰 STEP 1: Collecting article URLs...\n")
    
    for category in CATEGORIES:
        print(f"🗂  Category: {category}")
        links = get_article_links(category, max_pages=max_pages_per_category)
        # Limit per category
        links = links[:max_articles_per_category]
        all_article_urls.extend(links)
        polite_sleep()
    
    # Remove duplicates
    all_article_urls = list(set(all_article_urls))
    print(f"\n✅ Total unique articles to scrape: {len(all_article_urls)}\n")
    
    # Save article list
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_article_urls, f, ensure_ascii=False, indent=2)
    
    # ── Step 2: Scrape comments ──
    print("💬 STEP 2: Scraping comments...\n")
    
    for i, url in enumerate(tqdm(all_article_urls, desc="Articles")):
        comments = scrape_article_comments(url)
        all_comments.extend(comments)
        
        # Save checkpoint every 20 articles
        if (i + 1) % 20 == 0:
            df_checkpoint = pd.DataFrame(all_comments)
            df_checkpoint.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"\n  💾 Checkpoint saved: {len(all_comments)} comments so far")
        
        polite_sleep()
    
    # ── Step 3: Save final dataset ──
    print("\n📊 STEP 3: Saving dataset...\n")
    
    df = pd.DataFrame(all_comments)
    
    if df.empty:
        print("⚠️  No comments found. The site structure may have changed.")
        print("    Try running: python inspect_hespress.py")
        return
    
    # Basic cleaning
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= 10]
    df = df.reset_index(drop=True)
    
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print(f"✅ Dataset saved: {OUTPUT_FILE}")
    print(f"📈 Total comments collected: {len(df)}")
    print(f"📰 From articles: {df['article_url'].nunique()}")
    print(f"🗂  Categories: {df['category'].value_counts().to_dict()}")
    print(f"\n📋 Sample comments:")
    print(df["text"].head(5).to_string())
    
    return df


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_scraper(
        max_articles_per_category=30,  # Increase to 50-100 for more data
        max_pages_per_category=3        # Increase for more articles
    )
