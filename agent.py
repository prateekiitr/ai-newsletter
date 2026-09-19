"""
╔══════════════════════════════════════════════════════════════════╗
║  Daily AI Newsletter Agent — News + Papers                      ║
║  Dr. Prateek Singh · prateeksinghphd.in                         ║
║                                                                  ║
║  Two sections only: genuinely interesting AI news, and new      ║
║  papers worth reading. Candidates are fetched + keyword/recency ║
║  scored as before, then an LLM pass curates down to the items   ║
║  that are actually significant (not just keyword matches).      ║
╚══════════════════════════════════════════════════════════════════╝

ENV variables:
    GROQ_API_KEY, RESEND_API_KEY, ADMIN_TOKEN, SUBSCRIBERS_URL,
    FROM_EMAIL, FROM_NAME, TEST_MODE, TEST_EMAIL
"""

import os
import json
import time
import logging
import hashlib
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Config
GROQ_API_KEY    = os.environ['GROQ_API_KEY']
RESEND_API_KEY  = os.environ['RESEND_API_KEY']
ADMIN_TOKEN     = os.environ['ADMIN_TOKEN']
SUBSCRIBERS_URL = os.environ.get('SUBSCRIBERS_URL', 'https://prateeksinghphd.in/api/subscribers')
FROM_EMAIL      = os.environ.get('FROM_EMAIL', 'hello@prateeksinghphd.in')
FROM_NAME       = os.environ.get('FROM_NAME', 'Dr. Prateek Singh')
TEST_MODE       = os.environ.get('TEST_MODE', 'false').lower() == 'true'
TEST_EMAIL      = os.environ.get('TEST_EMAIL', 'prateek29singh@gmail.com')

# ── Deduplication with expiry (14 days)
SEEN_FILE = 'seen_titles_expiry.json'
TITLE_EXPIRE_DAYS = 14

# ── Keywords (high priority only – no generic ML)
HIGH_PRIORITY_KEYWORDS = [
    'large language model', 'llm', 'gpt', 'claude', 'gemini', 'llama', 'mistral',
    'qwen', 'deepseek', 'phi', 'gemma',
    'ai agent', 'agentic', 'openclaw', 'nemoclaw', 'hermes agent',
    'multi-agent', 'tool use', 'function calling', 'mcp', 'agent harness',
    'on-device', 'edge ai', 'edge inference', 'mobile ai',
    'quantization', 'llm inference', 'gguf', 'qlora', 'lora',
    'npu', 'qualcomm', 'snapdragon', 'samsung ai', 'tflite', 'onnx',
    'reasoning model', 'chain of thought', 'rlhf', 'grpo', 'dpo',
    'mamba', 'ssm', 'mixture of experts', 'moe', 'kv cache',
    'vllm', 'tensorrt', 'triton', 'speculative decoding', 'flash attention',
    # News‑specific triggers
    'release', 'announce', 'launch', 'new model', 'updated',
    # Industry drama / buzz triggers (funding, rivalry, controversy — not just technical releases)
    'lawsuit', 'sues', 'sued', 'fired', 'resigns', 'resignation', 'steps down',
    'raises', 'funding round', 'valuation', 'acquire', 'acquisition', 'ipo',
    'backlash', 'controversy', 'security breach', 'hacked', 'leaked', 'banned',
    'fine', 'regulation', 'antitrust', 'feud', 'rivalry', 'poach', 'departs',
    'ceo', 'billion', 'safety concern', 'whistleblower'
]

# ── SOURCES (only high‑churn, daily‑changing feeds)
# category 'paper' = research papers section, 'news' = genuine AI news section
SOURCES = [
    # Papers
    {'name': 'ArXiv cs.CL', 'url': 'https://rss.arxiv.org/rss/cs.CL', 'type': 'rss', 'priority': 1, 'category': 'paper'},
    {'name': 'ArXiv cs.AI', 'url': 'https://rss.arxiv.org/rss/cs.AI', 'type': 'rss', 'priority': 1, 'category': 'paper'},
    {'name': 'HuggingFace Daily Papers', 'type': 'hf_papers', 'priority': 1, 'category': 'paper'},

    # News — primary labs (official releases, tends dry/technical)
    {'name': 'OpenAI Blog', 'url': 'https://openai.com/blog/rss.xml', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'Anthropic News', 'url': 'https://www.anthropic.com/rss.xml', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'Google DeepMind Blog', 'url': 'https://deepmind.google/blog/rss.xml', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'Meta AI Blog', 'url': 'https://ai.meta.com/blog/rss/', 'type': 'rss', 'priority': 2, 'category': 'news'},
    {'name': 'Mistral AI Blog', 'url': 'https://mistral.ai/feed', 'type': 'rss', 'priority': 2, 'category': 'news'},

    # News — industry press (funding, rivalry, controversy, drama — the "interesting" stuff)
    {'name': 'TechCrunch AI', 'url': 'https://techcrunch.com/category/artificial-intelligence/feed/', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'VentureBeat AI', 'url': 'https://venturebeat.com/category/ai/feed/', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'The Verge AI', 'url': 'https://www.theverge.com/rss/ai-artificial-intelligence/index.xml', 'type': 'rss', 'priority': 1, 'category': 'news'},

    # News — "what's trending" approximation for X/LinkedIn (no official free API for either;
    # these aggregators surface the same stories once they go viral on social)
    {'name': 'Techmeme', 'url': 'https://www.techmeme.com/feed.xml', 'type': 'rss', 'priority': 1, 'category': 'news'},
    {'name': 'Google News — AI', 'url': 'https://news.google.com/rss/search?q=artificial%20intelligence%20when:2d&hl=en-US&gl=US&ceid=US:en', 'type': 'rss', 'priority': 1, 'category': 'news'},

    # News — dynamic sources for daily variety
    {'name': 'GitHub Trending (AI/LLM)', 'type': 'github_trending', 'priority': 2, 'category': 'news'},
    {'name': 'Reddit r/LocalLLaMA', 'type': 'reddit', 'subreddit': 'LocalLLaMA', 'priority': 2, 'category': 'news'},
    {'name': 'Reddit r/MachineLearning', 'type': 'reddit', 'subreddit': 'MachineLearning', 'priority': 3, 'category': 'news'},

    # News — Hacker News (technical query + a broad one to catch drama/funding/viral stories)
    {'name': 'Hacker News — LLM', 'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=LLM+language+model&hitsPerPage=6&numericFilters=created_at_i>{}', 'type': 'hn_api', 'priority': 2, 'category': 'news'},
    {'name': 'Hacker News — AI Agents', 'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=AI+agent+Claude+OpenAI&hitsPerPage=6&numericFilters=created_at_i>{}', 'type': 'hn_api', 'priority': 2, 'category': 'news'},
    {'name': 'Hacker News — AI Buzz', 'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=OpenAI+Anthropic+AI&hitsPerPage=10&numericFilters=created_at_i>{},points>30', 'type': 'hn_api', 'priority': 1, 'category': 'news'},
]

MAX_NEWS_CANDIDATES = 20   # sent to LLM for curation
MAX_PAPER_CANDIDATES = 15  # sent to LLM for curation
MAX_NEWS_FINAL = 5         # picked by LLM for the email
MAX_PAPERS_FINAL = 4       # picked by LLM for the email

# ============================================================================
# DEDUPLICATION WITH EXPIRY
# ============================================================================

def load_seen_titles() -> dict:
    try:
        with open(SEEN_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_seen_titles(seen: dict):
    now = time.time()
    expiry_sec = TITLE_EXPIRE_DAYS * 86400
    fresh = {h: ts for h, ts in seen.items() if now - ts < expiry_sec}
    with open(SEEN_FILE, 'w') as f:
        json.dump(fresh, f)

def title_hash(title: str, url: str) -> str:
    key = (title.lower().strip() + url).encode('utf-8')
    return hashlib.md5(key).hexdigest()

# ============================================================================
# FETCHERS (RSS, HF, HN, GitHub, Reddit)
# ============================================================================

def fetch_rss(source: dict) -> list[dict]:
    try:
        feed = feedparser.parse(source['url'])
        items = []
        for entry in feed.entries[:10]:
            # Compute approximate age (if published date exists)
            days_old = 999
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub = time.mktime(entry.published_parsed)
                days_old = (time.time() - pub) / 86400
            items.append({
                'title':    entry.get('title', '').strip(),
                'summary':  entry.get('summary', entry.get('description', ''))[:500].strip(),
                'url':      entry.get('link', ''),
                'source':   source['name'],
                'priority': source['priority'],
                'category': source['category'],
                'days_old': days_old
            })
        log.info(f"  {source['name']}: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  RSS fetch failed for {source['name']}: {e}")
        return []

def fetch_hf_papers(source: dict) -> list[dict]:
    try:
        r = requests.get('https://huggingface.co/api/daily_papers', timeout=10)
        papers = r.json()
        items = []
        for p in papers[:10]:
            paper = p.get('paper', {})
            items.append({
                'title':    paper.get('title', '').strip(),
                'summary':  paper.get('summary', '')[:500].strip(),
                'url':      f"https://huggingface.co/papers/{paper.get('id', '')}",
                'source':   'HuggingFace Papers',
                'priority': source['priority'],
                'category': source['category'],
                'days_old': 0  # fresh daily
            })
        log.info(f"  HuggingFace Papers: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  HF papers fetch failed: {e}")
        return []

def fetch_hn(source: dict) -> list[dict]:
    try:
        yesterday = int(time.time()) - 86400
        url = source['url'].format(yesterday)
        r = requests.get(url, timeout=10)
        data = r.json()
        items = []
        for hit in data.get('hits', []):
            items.append({
                'title':    hit.get('title', '').strip(),
                'summary':  f"Points: {hit.get('points', 0)} | Comments: {hit.get('num_comments', 0)}",
                'url':      hit.get('url') or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                'source':   source['name'],
                'priority': source['priority'],
                'category': source['category'],
                'days_old': 0
            })
        log.info(f"  {source['name']}: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  HN fetch failed ({source['name']}): {e}")
        return []

def fetch_github_trending(source: dict) -> list[dict]:
    """Fetch recently-popular AI/LLM repos via the official GitHub Search API
    (avoids relying on unofficial trending mirrors, which go down often)."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
        url = (
            "https://api.github.com/search/repositories"
            f"?q=topic:llm+created:>{since}&sort=stars&order=desc&per_page=8"
        )
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'AI-Newsletter/1.0'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = []
        for repo in data.get('items', [])[:8]:
            name = repo.get('full_name', '')
            description = repo.get('description', '') or "No description"
            items.append({
                'title': f"⭐ {repo.get('stargazers_count', 0)} stars · {name}",
                'summary': description[:400],
                'url': repo.get('html_url', '#'),
                'source': 'GitHub Trending (AI/LLM)',
                'priority': source['priority'],
                'category': source['category'],
                'days_old': 0
            })
        log.info(f"  GitHub Trending: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"GitHub trending fetch failed: {e}")
        return []

def fetch_reddit(source: dict) -> list[dict]:
    subreddit = source.get('subreddit', 'LocalLLaMA')
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=8"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; AI-Newsletter/1.0)'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        items = []
        for post in data['data']['children']:
            p = post['data']
            items.append({
                'title': p['title'],
                'summary': (p.get('selftext', '') or f"↑{p['score']} · {p['num_comments']} comments")[:400],
                'url': f"https://reddit.com{p['permalink']}",
                'source': f"Reddit r/{subreddit}",
                'priority': source['priority'],
                'category': source['category'],
                'days_old': 0
            })
        log.info(f"  Reddit r/{subreddit}: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"Reddit fetch failed for r/{subreddit}: {e}")
        return []

# ============================================================================
# SCORING (No health block, recency bonus)
# ============================================================================

def score_item(item: dict) -> float:
    text = (item['title'] + ' ' + item['summary']).lower()
    score = 0.0
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in text:
            score += 5.0
    # Recency bonus: items from today or yesterday get +2
    if item.get('days_old', 999) <= 1:
        score += 2.0
    elif item.get('days_old', 999) <= 2:
        score += 1.0
    # Priority penalty (lower priority sources get slight reduction)
    score -= (item['priority'] - 1) * 0.5
    return score

# ============================================================================
# FETCH ALL NEWS (with dedup, expiry, diversity)
# ============================================================================

def _diversify(items: list[dict], max_per_source: int, cap: int) -> list[dict]:
    source_count = {}
    out = []
    for item in items:
        src = item['source']
        if source_count.get(src, 0) < max_per_source:
            out.append(item)
            source_count[src] = source_count.get(src, 0) + 1
        if len(out) >= cap:
            break
    return out

def fetch_all() -> tuple[list[dict], list[dict]]:
    """Fetch every source and split scored, deduped candidates into (news, papers)."""
    log.info("Fetching news and papers from all sources...")
    all_items = []

    for source in SOURCES:
        if source['type'] == 'rss':
            all_items.extend(fetch_rss(source))
        elif source['type'] == 'hf_papers':
            all_items.extend(fetch_hf_papers(source))
        elif source['type'] == 'hn_api':
            all_items.extend(fetch_hn(source))
        elif source['type'] == 'github_trending':
            all_items.extend(fetch_github_trending(source))
        elif source['type'] == 'reddit':
            all_items.extend(fetch_reddit(source))
        time.sleep(0.3)  # polite rate limiting

    # Score each item
    for item in all_items:
        item['score'] = score_item(item)

    # Load seen titles (expiry handled inside save/load)
    seen = load_seen_titles()
    now = time.time()

    # Sort by score descending, then deduplicate & filter seen
    unique_items = []
    for item in sorted(all_items, key=lambda x: x['score'], reverse=True):
        if not item['title']:
            continue
        h = title_hash(item['title'], item['url'])
        if h in seen:
            log.debug(f"Skipping seen: {item['title'][:60]}")
            continue
        seen[h] = now  # mark as seen now (persisted at end)
        unique_items.append(item)

    news_pool = [i for i in unique_items if i['category'] == 'news']
    paper_pool = [i for i in unique_items if i['category'] == 'paper']

    news_candidates = _diversify(news_pool, max_per_source=3, cap=MAX_NEWS_CANDIDATES)
    paper_candidates = _diversify(paper_pool, max_per_source=8, cap=MAX_PAPER_CANDIDATES)

    save_seen_titles(seen)

    log.info(f"Candidates for LLM curation: {len(news_candidates)} news, {len(paper_candidates)} papers")
    return news_candidates, paper_candidates

# ============================================================================
# LLM curation, subscriber fetch, email rendering/sending, main
# ============================================================================

def llm_curate_digest(news_candidates: list[dict], paper_candidates: list[dict], date_str: str) -> dict:
    """Ask the LLM to judge which candidates are genuinely interesting — not just keyword hits —
    and write the digest from the ones it picks."""
    def fmt(items):
        return '\n\n'.join([
            f"[{i+1}] SOURCE: {item['source']}\nTITLE: {item['title']}\nSUMMARY: {item['summary'][:220]}\nURL: {item['url']}"
            for i, item in enumerate(items)
        ])

    news_text = fmt(news_candidates)
    papers_text = fmt(paper_candidates)

    system_prompt = f"""You are curating and writing a daily AI newsletter for Dr. Prateek Singh,
Staff Engineer/Manager on the Compute AI team at Qualcomm.
IIT Roorkee PhD. Expert in LLM deployment, on-device AI, quantization,
AI agents, and edge inference.

You are given two candidate pools already fetched from RSS/HN/Reddit/GitHub/arXiv/HuggingFace/
Techmeme/Google News (Techmeme and Google News track stories once they go viral on X/Twitter and
LinkedIn, so treat items from them as a proxy for "what's trending on social" even though we don't
hit those platforms directly).
Your job is not just to summarize them — it's to CURATE. Most candidates will be routine,
repetitive, or low-signal. Skip anything that is hype, a rehash, a minor point release, or a
low-effort post — even if it matches AI keywords.

The reader finds pure "Model X released, here are the benchmarks" posts boring even when they're
technically significant. For the "news" section, actively prefer items that are genuinely
INTERESTING, not just technically important. That means favoring, when available:
  - Industry drama: lawsuits, executive departures/hires, public feuds, whistleblowers, leaks
  - Competitive moves: one lab undercutting/copying/reacting to another, rivalry, poaching
  - Money: funding rounds, valuations, acquisitions, layoffs
  - Controversy: safety incidents, security breaches, backlash, bans, regulation fights
  - Odd/funny/unexpected happenings in the AI world — the stuff people actually talk about
A big model release is worth including only if it's genuinely notable (a new frontier model, a
surprising capability, a real shift) — not every routine update. If the news pool this run is all
dry lab-blog announcements, say so plainly in the closing thought rather than padding with filler.

VOICE: Confident, clear, slightly technical but accessible. Not hype-y.
Write like a senior engineer who has seen a lot of AI trends come and go.
Short sentences. No fluff. Respect the reader's time.

FOCUS for papers only: LLMs, AI agents, model releases, inference optimization, open-source AI,
on-device AI, quantization, agent frameworks, reasoning models.

OUTPUT FORMAT — return valid JSON only, no markdown, no explanation:
{{
  "subject": "email subject line (max 70 chars, include date and 1-2 key topics)",
  "news": [
    {{"headline": "one punchy line", "body": "2-3 sentences on what happened and why it matters", "url": "url from the news candidates"}}
  ],
  "papers": [
    {{"title": "short title", "summary": "one sentence — what it does and why it matters", "url": "url from the paper candidates"}}
  ],
  "closing_thought": "1 short sentence — an honest observation or provocative question about today's AI landscape. No positivity fluff."
}}

You MUST return at least 3 "news" items and at least 3 "papers" items every time (up to
{MAX_NEWS_FINAL} news and {MAX_PAPERS_FINAL} papers) — the email layout has dedicated sections for
both and an empty or near-empty section looks broken. The paper pool is pre-filtered to arXiv/
HuggingFace research, so it will always have at least 3 legitimate candidates; do not return fewer
just because none of them feel like "big" news — pick the most relevant/interesting ones you do
have. Only go below 3 in a section if that candidate list above is truly shorter than 3 items.
Never invent items or urls not present in the candidates below."""

    user_prompt = f"""Date: {date_str}

NEWS CANDIDATES:
{news_text}

PAPER CANDIDATES:
{papers_text}

Return only the JSON object. No markdown. No explanation."""

    log.info("Calling Groq API...")
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }

    models_to_try = [
        ('openai/gpt-oss-120b',      True),
        ('openai/gpt-oss-20b',       True),
    ]

    last_error = None
    for model_name, supports_json in models_to_try:
        try:
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 3000,
            }
            if supports_json:
                payload['response_format'] = {'type': 'json_object'}

            log.info(f"  Trying model: {model_name}")
            r = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            if not r.ok:
                log.warning(f"  Groq {r.status_code} body: {r.text[:500]}")
            r.raise_for_status()
            content = r.json()['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()
            digest = json.loads(content)
            log.info(f"Digest curated with {model_name}. Subject: {digest.get('subject', 'N/A')} "
                     f"({len(digest.get('news', []))} news, {len(digest.get('papers', []))} papers)")
            return digest
        except Exception as e:
            log.warning(f"  Model {model_name} failed: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

def fetch_subscribers() -> list[str]:
    if TEST_MODE:
        log.info(f"TEST MODE — using test email: {TEST_EMAIL}")
        return [TEST_EMAIL]
    log.info("Fetching subscribers from Cloudflare...")
    try:
        url = SUBSCRIBERS_URL
        if 'token=' not in url:
            url = f"{SUBSCRIBERS_URL}?token={ADMIN_TOKEN}"
        r = requests.get(url, headers={'Authorization': f'Bearer {ADMIN_TOKEN}'}, timeout=15)
        log.info(f"Subscriber API status: {r.status_code}")
        r.raise_for_status()
        data = r.json()
        all_subs = data.get('subscribers', [])
        emails = [s['email'] for s in all_subs if s.get('active', True) is not False and s.get('email')]
        log.info(f"Active subscribers: {len(emails)}")
        return emails
    except Exception as e:
        log.error(f"Failed to fetch subscribers: {e}")
        raise

def build_email_html(digest: dict, date_str: str, email: str) -> str:
    unsubscribe_url = f"https://prateeksinghphd.in/api/unsubscribe?email={requests.utils.quote(email)}"

    news_html = ''
    for i, n in enumerate(digest.get('news', [])[:MAX_NEWS_FINAL]):
        border_top = 'border-top:1px solid #33333f;' if i > 0 else ''
        news_html += f"""
        <div style="padding:22px 0;{border_top}">
          <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:8px;">
            <span style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:22px;font-weight:900;
                         color:#00d9b4;opacity:.35;line-height:1.2;">0{i+1}</span>
            <a href="{n.get('url','#')}"
               style="font-size:19px;font-weight:700;color:#f0f0f8;
                      text-decoration:none;line-height:1.3;">{n.get('headline','')}</a>
          </div>
          <p style="color:#b0b0c8;font-size:16px;line-height:1.75;margin:0 0 10px;
                    padding-left:34px;">{n.get('body','')}</p>
          <div style="padding-left:34px;">
            <a href="{n.get('url','#')}"
               style="font-family:monospace;font-size:11px;letter-spacing:2px;
                      color:#00d9b4;text-transform:uppercase;text-decoration:none;">
              Read →
            </a>
          </div>
        </div>"""

    paper_colors = ['#7c6bff', '#ff6b9d', '#00d9b4', '#ffb347']
    papers_html = ''
    for i, p in enumerate(digest.get('papers', [])[:MAX_PAPERS_FINAL]):
        color = paper_colors[i % len(paper_colors)]
        papers_html += f"""
        <div style="margin-bottom:20px;background:#232330;border:1px solid #33333f;
                    border-left:3px solid {color};border-radius:6px;padding:20px 24px;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <span style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:28px;font-weight:900;
                         color:{color};opacity:.35;line-height:1;">0{i+1}</span>
            <a href="{p.get('url','#')}"
               style="font-size:17px;font-weight:700;color:#f0f0f8;
                      text-decoration:none;line-height:1.35;">{p.get('title','')}</a>
          </div>
          <p style="color:#9090b8;font-size:15px;line-height:1.7;margin:0 0 14px;
                    padding-left:44px;">{p.get('summary','')}</p>
          <div style="padding-left:44px;">
            <a href="{p.get('url','#')}"
               style="display:inline-block;background:{color}18;color:{color};
                      border:1px solid {color}40;font-family:monospace;font-size:11px;
                      letter-spacing:2px;text-transform:uppercase;text-decoration:none;
                      padding:6px 14px;border-radius:3px;">
              Read Paper →
            </a>
          </div>
        </div>"""

    closing = digest.get('closing_thought', '')
    subject = digest.get('subject', f'AI Daily — {date_str}')
    try:
        dow = datetime.now().strftime('%A').upper()
    except Exception:
        dow = 'TODAY'
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>{subject}</title></head>
<body style="margin:0;padding:0;background:#1a1a22;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
<div style="max-width:700px;margin:0 auto;background:#1a1a22;">
  <div style="background:#00d9b4;padding:10px 40px;text-align:center;">
    <span style="font-family:monospace;font-size:11px;font-weight:700;letter-spacing:3px;color:#08080f;text-transform:uppercase;">
      🧠 AI DAILY BRIEFING &nbsp;·&nbsp; {dow} &nbsp;·&nbsp; 9 AM IST
    </span>
  </div>
  <div style="padding:40px 44px 32px;border-bottom:1px solid #33333f;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr><td><p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px;font-style:italic;color:#00d9b4;margin:0 0 6px;">Dr. Prateek Singh</p>
          <h1 style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:36px;font-weight:900;color:#f0f0f8;margin:0;line-height:1.1;letter-spacing:-0.5px;">Your AI Briefing</h1>
          <p style="font-size:15px;color:#5a5a7a;margin:8px 0 0;font-family:monospace;">{date_str}</p></td>
        <td style="text-align:right;vertical-align:top;padding-top:4px;"><div style="background:#232330;border:1px solid #33333f;border-radius:6px;padding:10px 16px;display:inline-block;">
          <p style="font-family:monospace;font-size:10px;letter-spacing:2px;color:#5a5a7a;text-transform:uppercase;margin:0 0 3px;">Today</p>
          <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:22px;font-weight:900;color:#00d9b4;margin:0;line-height:1;">5 min</p>
          <p style="font-family:monospace;font-size:9px;color:#3a3a5a;margin:2px 0 0;text-transform:uppercase;">read</p>
        </div></td></tr>
    </table>
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #33333f;">
    <div style="display:inline-block;background:#00d9b414;border:1px solid #00d9b430;border-radius:3px;padding:4px 12px;margin-bottom:8px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#00d9b4;text-transform:uppercase;font-weight:700;">📰 Genuinely Interesting AI News</span>
    </div>
    {news_html}
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #33333f;background:#20202a;">
    <div style="display:inline-block;background:#7c6bff14;border:1px solid #7c6bff30;border-radius:3px;padding:4px 12px;margin-bottom:22px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#7c6bff;text-transform:uppercase;font-weight:700;">📄 New Papers Worth Reading</span>
    </div>
    {papers_html}
  </div>
  <div style="padding:30px 44px;border-bottom:1px solid #33333f;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr><td style="width:4px;background:linear-gradient(180deg,#00d9b4,#7c6bff);border-radius:2px;">&nbsp;</td>
    <td style="padding-left:20px;"><p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:17px;font-style:italic;color:#8a8aaa;line-height:1.75;margin:0;">"{closing}"</p>
    <p style="font-size:13px;color:#4a4a6a;margin:10px 0 0;font-family:monospace;">— Dr. Prateek Singh</p></td></tr></table>
  </div>
  <div style="padding:32px 44px;background:#232330;border-bottom:1px solid #33333f;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr><td style="vertical-align:middle;">
      <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:18px;font-weight:700;color:#f0f0f8;margin:0 0 4px;">Building with LLMs or AI Agents?</p>
      <p style="font-size:14px;color:#6a6a8a;margin:0;">Let's discuss your project — free 30-min call.</p></td>
    <td style="text-align:right;vertical-align:middle;"><a href="https://cal.com/prateek-singh-la8jpj" style="display:inline-block;background:#00d9b4;color:#08080f;font-family:monospace;font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase;text-decoration:none;padding:12px 20px;border-radius:3px;white-space:nowrap;">Book a Call →</a></td></tr></table>
  </div>
  <div style="padding:32px 44px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;"><tr><td>
      <p style="font-family:monospace;font-size:10px;letter-spacing:2px;color:#00d9b4;text-transform:uppercase;margin:0 0 8px;">Dr. Prateek Singh</p>
      <p style="font-size:13px;color:#4a4a6a;line-height:1.8;margin:0;">Staff Engineer/Manager, Compute AI<br>Qualcomm · IIT Roorkee PhD</p></td>
    <td style="text-align:right;vertical-align:middle;"><a href="https://prateeksinghphd.in" style="display:inline-block;background:#232330;color:#00d9b4;border:1px solid #33333f;font-family:monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;text-decoration:none;padding:8px 14px;border-radius:3px;">My Blog →</a></td></tr></table>
    <div style="border-top:1px solid #33333f;padding-top:20px;">
      <a href="https://prateeksinghphd.in" style="font-size:13px;color:#4a4a6a;text-decoration:none;margin-right:16px;">🌐 Website</a>
      <a href="https://www.linkedin.com/in/prateek29s/" style="font-size:13px;color:#4a4a6a;text-decoration:none;margin-right:16px;">💼 LinkedIn</a>
      <a href="https://scholar.google.com/citations?user=nYZhJaMAAAAJ&hl=en" style="font-size:13px;color:#4a4a6a;text-decoration:none;margin-right:16px;">📚 Scholar</a>
      <a href="https://cal.com/prateek-singh-la8jpj" style="font-size:13px;color:#4a4a6a;text-decoration:none;">📅 Book a Call</a>
    </div>
    <p style="margin:24px 0 0;"><a href="{unsubscribe_url}" style="font-family:monospace;font-size:10px;letter-spacing:2px;color:#2a2a4a;text-decoration:none;text-transform:uppercase;">Unsubscribe · One click, no questions asked</a></p>
  </div>
</div>
</body>
</html>"""

def send_newsletter(emails: list[str], digest: dict, date_str: str) -> dict:
    results = {'sent': 0, 'failed': 0, 'errors': []}
    log.info(f"Sending to {len(emails)} subscribers...")
    for i, email in enumerate(emails):
        try:
            html = build_email_html(digest, date_str, email)
            subject = digest.get('subject', f'🧠 AI Daily — {date_str}')
            r = requests.post(
                'https://api.resend.com/emails',
                headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
                json={'from': f'{FROM_NAME} <{FROM_EMAIL}>', 'to': email, 'subject': subject, 'html': html},
                timeout=15
            )
            if r.status_code == 200:
                results['sent'] += 1
                if (i + 1) % 10 == 0:
                    log.info(f"  Sent {i+1}/{len(emails)}...")
            else:
                results['failed'] += 1
                results['errors'].append({'email': email, 'status': r.status_code, 'body': r.text[:100]})
                log.warning(f"  Failed for {email}: {r.status_code}")
            time.sleep(0.6)
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({'email': email, 'error': str(e)})
            log.error(f"  Exception for {email}: {e}")
    return results

def main():
    date_str = datetime.now(timezone.utc).strftime('%B %d, %Y')
    log.info(f"{'='*60}")
    log.info(f"Daily AI Newsletter Agent — {date_str}")
    if TEST_MODE:
        log.info("⚠️  TEST MODE — emails only sent to TEST_EMAIL")
    log.info(f"{'='*60}")

    news_candidates, paper_candidates = fetch_all()
    if not news_candidates and not paper_candidates:
        log.error("No candidates fetched — aborting")
        return

    digest = llm_curate_digest(news_candidates, paper_candidates, date_str)
    if not digest.get('news') and not digest.get('papers'):
        log.error("LLM curated zero items — aborting")
        return

    emails = fetch_subscribers()
    if not emails:
        log.error("No subscribers found — aborting")
        return

    results = send_newsletter(emails, digest, date_str)

    log.info(f"{'='*60}")
    log.info(f"✅ Sent:   {results['sent']}")
    log.info(f"❌ Failed: {results['failed']}")
    if results['errors']:
        log.warning(f"Errors: {json.dumps(results['errors'], indent=2)}")
    log.info(f"{'='*60}")

if __name__ == '__main__':
    main()
