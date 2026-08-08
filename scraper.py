import asyncio
import os
import aiohttp
import asyncpg
import feedparser
import trafilatura
import httpx
import numpy as np
import urllib.parse
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from curl_cffi.requests import AsyncSession
import instructor
from openai import AsyncOpenAI

# Suppress harmless HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# CONFIGURATION
# ==========================================
SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']
SUPABASE_DB_URL = os.environ['SUPABASE_DB_URL']

OMNIROUTE_API_KEY = os.environ['OMNIROUTE_API_KEY'] 
OMNIROUTE_BASE_URL = os.environ['OMNIROUTE_BASE_URL']

# Semaphores for concurrent rate limiting
SCRAPE_SEMAPHORE = asyncio.Semaphore(10)
LLM_SEMAPHORE = asyncio.Semaphore(2)

GOOGLE_QUERY = urllib.parse.quote("Salesforce AND (Agentforce OR \"Data Cloud\" OR layoffs OR acquisition OR CTA OR certification)")
GOOGLE_NEWS_URL = f"https://news.google.com/rss/search?q={GOOGLE_QUERY}&hl=en-US&gl=US&ceid=US:en"

RSS_FEEDS = [
    GOOGLE_NEWS_URL,
    "https://developer.salesforce.com/blogs/feed",
    "https://www.salesforceben.com/feed/",
    "https://admin.salesforce.com/feed",
    "https://www.reddit.com/r/salesforce/new/.rss",
    "https://www.reddit.com/r/salesforce_developers/new/.rss",
    "https://www.reddit.com/r/SalesforceCareers/new/.rss"
]

print("Loading CPU Vector Embedding Engine...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Configure HTTP client with a wide 60s timeout so the 45s wait_for acts as the strict enforcer
TIMEOUT_CONFIG = httpx.Timeout(60.0, connect=10.0)

# Initialize the SINGLE OmniRoute Client
omni_client = AsyncOpenAI(
    base_url=OMNIROUTE_BASE_URL,
    api_key=OMNIROUTE_API_KEY,
    http_client=httpx.AsyncClient(timeout=TIMEOUT_CONFIG)
)
ai_client = instructor.from_openai(omni_client, mode=instructor.Mode.JSON)

# ==========================================
# PYDANTIC STRICT SCHEMA
# ==========================================
class SalesforceBrief(BaseModel):
    is_signal: bool = Field(
        ..., 
        description="True ONLY if text covers high-impact news: Agentforce, Data Cloud, M&A, layoffs, salary/market trends, CTA architecture, or free certs. False for tutorials, basic homework, or marketing fluff."
    )
    headline: Optional[str] = Field(None, description="Clear, professional headline summarizing what actually happened.")
    event_announcement: Optional[str] = Field(None, description="2 concise sentences detailing the core factual event.")
    system_architecture_impact: Optional[str] = Field(None, description="Technical impact explained in a clear, beginner-friendly format. Explain what systems change and *why* it matters, avoiding overly advanced jargon so the reader can learn.")
    enterprise_advisory_strategy: Optional[str] = Field(None, description="Consulting perspective detailing strategic takeaways for corporate clients, potential new billable use cases, or demo opportunities.")

# ==========================================
# SUPABASE POSTGRES DB LAYER (WITH POOLING)
# ==========================================
def format_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url

async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                url TEXT PRIMARY KEY,
                title TEXT,
                embedding BYTEA,
                processed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

async def get_historical_vectors(pool):
    async with pool.acquire() as conn:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        rows = await conn.fetch("SELECT url, embedding FROM articles WHERE processed_at > $1", seven_days_ago)
        
        seen_urls = {row['url'] for row in rows}
        vectors = []
        for row in rows:
            if row['embedding']:
                vectors.append(np.frombuffer(row['embedding'], dtype=np.float32))
        
        return seen_urls, np.array(vectors) if vectors else None

async def save_article_to_db(pool, url, title, embedding):
    async with pool.acquire() as conn:
        embedding_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else None
        await conn.execute("""
            INSERT INTO articles (url, title, embedding, processed_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (url) DO UPDATE 
            SET title = EXCLUDED.title, embedding = EXCLUDED.embedding, processed_at = EXCLUDED.processed_at;
        """, url, title, embedding_bytes, datetime.now(timezone.utc))

# ==========================================
# VECTOR DEDUPLICATION & CLUSTERING
# ==========================================
def filter_and_cluster_articles(new_articles, historical_vectors, distance_threshold=0.25):
    if not new_articles:
        return []

    titles = [a['title'] for a in new_articles]
    new_embeddings = embedder.encode(titles, normalize_embeddings=True)

    candidates = []
    candidate_embeddings = []

    for idx, article in enumerate(new_articles):
        emb = new_embeddings[idx]
        article['embedding'] = emb
        
        if historical_vectors is not None and len(historical_vectors) > 0:
            similarities = np.dot(historical_vectors, emb)
            max_sim = np.max(similarities) if len(similarities) > 0 else 0
            if max_sim >= (1 - distance_threshold):
                print(f"-> Skipped (Matches recent news in Supabase): {article['title'][:40]}...")
                continue
        
        candidates.append(article)
        candidate_embeddings.append(emb)

    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering.fit_predict(np.array(candidate_embeddings))

    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(candidates[idx])

    representatives = [max(items, key=lambda x: len(x.get('summary', ''))) for items in clusters.values()]
    print(f"Batch Deduplication: Reduced {len(new_articles)} items to {len(representatives)} distinct clusters.")
    return representatives

# ==========================================
# EVASIVE WAF SCRAPING (curl_cffi)
# ==========================================
async def extract_full_text(url):
    async with SCRAPE_SEMAPHORE:
        try:
            async with AsyncSession(impersonate="chrome124", timeout=12) as session:
                resp = await session.get(url)
                if resp.status_code == 200:
                    text = trafilatura.extract(resp.content, include_comments=False, favor_precision=True)
                    if text and len(text) > 200:
                        return text[:4000]

                # Fallback: Jina AI Reader API
                jina_url = f"https://r.jina.ai/{url}"
                jina_resp = await session.get(jina_url)
                if jina_resp.status_code == 200 and len(jina_resp.text) > 200:
                    return jina_resp.text[:4000]
        except Exception:
            pass
        return None

# ==========================================
# INSTRUCTED LLM INFERENCE (Powered by OmniRoute Chaos)
# ==========================================
async def analyze_article(article):
    async with LLM_SEMAPHORE:
        full_text = await extract_full_text(article['link'])
        
        prompt = f"""
        Act as an Enterprise Technology Analyst & Salesforce CTA Strategist.
        Title: {article['title']}
        Link: {article['link']}
        Full Content: {full_text if full_text else article.get('summary', 'Title only')}
        """

        try:
            # 45-second strict timeout to give Chaos Engine time to cycle models
            response = await asyncio.wait_for(
                ai_client.chat.completions.create(
                    model="auto", 
                    response_model=SalesforceBrief,
                    max_retries=2, 
                    messages=[
                        {"role": "system", "content": "Extract hard technical, platform, and career signals. Ignore marketing fluff. No emojis."},
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=45.0
            )
            return article, response
        except Exception as e:
            print(f"OmniRoute Inference Error on '{article['title'][:30]}': {e}")
            return article, None

# ==========================================
# INGESTION & DELIVERY
# ==========================================
async def fetch_single_feed(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                content = await resp.text()
                feed = feedparser.parse(content)
                return [{
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.get('summary', '')[:300]
                } for entry in feed.entries[:5]]
    except Exception as e:
        print(f"Feed error on {url}: {e}")
    return []

async def fetch_all_rss_feeds():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SalesforceArchitectBot/5.0'}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch_single_feed(session, url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_items = []
        for res in results:
            if isinstance(res, list):
                valid_items.extend(res)
        return valid_items

async def send_to_slack(brief: SalesforceBrief, link: str):
    # 1. Safely handle potential None values to prevent Block Kit crashes
    safe_headline = brief.headline[:150] if brief.headline else "Salesforce Intelligence Update"
    tech_impact = brief.system_architecture_impact if brief.system_architecture_impact else "Not specified."
    advisory_strategy = brief.enterprise_advisory_strategy if brief.enterprise_advisory_strategy else "Not specified."

    # 2. Pure Block Kit Payload (No Attachments)
    block_kit_payload = {
        "text": f"New Update: {safe_headline}", # Now invisible in chat, ONLY used for mobile push notifications
        "blocks": [
            {
                "type": "context",
                "elements": [
                    {
                        "type": "plain_text",
                        "text": "Enterprise Intelligence Gateway",
                        "emoji": False
                    }
                ]
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": safe_headline,
                    "emoji": False
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Event / Announcement*\n{brief.event_announcement}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*System & Architecture Impact*\n{tech_impact}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Enterprise Advisory Strategy*\n{advisory_strategy}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Read Full Article",
                            "emoji": False
                        },
                        "url": link,
                        "style": "primary"
                    },
                    {
                        "type": "checkboxes",
                        "options": [
                            {
                                "text": {
                                    "type": "plain_text",
                                    "text": "Mark as Read",
                                    "emoji": False
                                },
                                "value": "is_read"
                            }
                        ],
                        "action_id": "mark_read_action"
                    }
                ]
            },
            {
                "type": "divider" # Physical separator at the bottom of the card!
            }
        ]
    }

    # 3. Capture and log the actual response from Slack to prevent silent failures
    async with aiohttp.ClientSession() as session:
        async with session.post(SLACK_WEBHOOK_URL, json=block_kit_payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"-> Slack Webhook Blocked: {resp.status} - {error_text}")
            else:
                print(f"-> Successfully Pushed Block Kit to Slack: {safe_headline[:40]}")

# ==========================================
# MAIN EXECUTION (WITH CONNECTION POOLING)
# ==========================================
async def main():
    print("Initializing Enterprise Intelligence Engine (Supabase + WAF Bypass + OmniRoute Chaos)...")
    
    clean_url = format_db_url(SUPABASE_DB_URL)
    
    # Establish a connection pool to strictly prevent database connection limits from crashing GitHub Actions
    async with asyncpg.create_pool(clean_url, min_size=1, max_size=5) as db_pool:
        await init_db(db_pool)
        
        seen_urls, historical_vectors = await get_historical_vectors(db_pool)
        
        # 1. Fetch Feeds Concurrently
        raw_articles = await fetch_all_rss_feeds()
        net_new = [a for a in raw_articles if a['link'] not in seen_urls]
        print(f"Ingested {len(raw_articles)} items. {len(net_new)} URLs are net-new.")
        
        if not net_new:
            print("Pipeline complete. No new URLs.")
            return

        # 2. Vector Deduplication against History & Current Batch
        unique_targets = filter_and_cluster_articles(net_new, historical_vectors)
        if not unique_targets:
            print("All new items were semantic duplicates. Pipeline complete.")
            return

        # 3. Process AI Tasks Concurrently with Return Exceptions Guard
        print(f"Processing {len(unique_targets)} unique targets through OmniRoute Cascade...")
        ai_tasks = [analyze_article(item) for item in unique_targets]
        results = await asyncio.gather(*ai_tasks, return_exceptions=True)

        # 4. Deliver & Persist to Supabase
        for res in results:
            if isinstance(res, Exception) or res is None:
                continue
            
            article, brief = res
            
            # CRITICAL FIX: If brief is None, it means the API errored out. 
            # We DO NOT save it to the database. It will be pulled and retried again in 15 mins.
            if brief is None:
                print(f"-> Errored/Timed Out (Will retry next run): {article['title'][:40]}")
                continue
            
            if brief.is_signal:
                await send_to_slack(brief, article['link'])
                print(f"-> Pushed to Slack: {article['title'][:40]}")
            else:
                print(f"-> Skipped by AI (Low value): {article['title'][:40]}")
            
            # Only save to DB if the AI successfully generated a response
            await save_article_to_db(db_pool, article['link'], article['title'], article.get('embedding'))

    print("Pipeline execution complete. Supabase state updated.")

if __name__ == "__main__":
    asyncio.run(main())
