"""
╔══════════════════════════════════════════════════════════════════╗
║  Daily AI Newsletter Agent (Refactored for Freshness)           ║
║  Dr. Prateek Singh · prateeksinghphd.in                         ║
║                                                                  ║
║  Changes:                                                        ║
║  - Health AI NOT blocked (user request)                         ║
║  - Added GitHub Trending, Reddit for daily variety              ║
║  - Dedup with 14-day expiry (different news every day)          ║
║  - Diversity filter: max 2 items per source                     ║
║  - Recency bonus in scoring                                     ║
║  - Recommended run time: 6 PM IST (captures full day's news)    ║
╚══════════════════════════════════════════════════════════════════╝

ENV variables (same as before):
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
from datetime import datetime, timezone
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
MODEL           = 'llama3-70b-8192'
MAX_ITEMS       = 8
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
    'release', 'announce', 'launch', 'new model', 'updated'
]

# ── SOURCES (only high‑churn, daily‑changing feeds)
SOURCES = [
    # Primary research & labs
    {'name': 'ArXiv cs.CL', 'url': 'https://rss.arxiv.org/rss/cs.CL', 'type': 'rss', 'priority': 1},
    {'name': 'ArXiv cs.AI', 'url': 'https://rss.arxiv.org/rss/cs.AI', 'type': 'rss', 'priority': 1},
    {'name': 'HuggingFace Daily Papers', 'type': 'hf_papers', 'priority': 1},
    {'name': 'OpenAI Blog', 'url': 'https://openai.com/blog/rss.xml', 'type': 'rss', 'priority': 1},
    {'name': 'Anthropic News', 'url': 'https://www.anthropic.com/rss.xml', 'type': 'rss', 'priority': 1},
    {'name': 'Google DeepMind Blog', 'url': 'https://deepmind.google/blog/rss.xml', 'type': 'rss', 'priority': 1},
    {'name': 'Meta AI Blog', 'url': 'https://ai.meta.com/blog/rss/', 'type': 'rss', 'priority': 2},
    {'name': 'Mistral AI Blog', 'url': 'https://mistral.ai/feed', 'type': 'rss', 'priority': 2},
    
    # Dynamic sources for daily variety
    {'name': 'GitHub Trending (AI/LLM)', 'type': 'github_trending', 'priority': 2},
    {'name': 'Reddit r/LocalLLaMA', 'type': 'reddit', 'subreddit': 'LocalLLaMA', 'priority': 2},
    {'name': 'Reddit r/MachineLearning', 'type': 'reddit', 'subreddit': 'MachineLearning', 'priority': 3},
    
    # Hacker News (two queries for breadth)
    {'name': 'Hacker News — LLM', 'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=LLM+language+model&hitsPerPage=6&numericFilters=created_at_i>{}', 'type': 'hn_api', 'priority': 2},
    {'name': 'Hacker News — AI Agents', 'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=AI+agent+Claude+OpenAI&hitsPerPage=6&numericFilters=created_at_i>{}', 'type': 'hn_api', 'priority': 2},
]

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
                'days_old': 0
            })
        log.info(f"  {source['name']}: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  HN fetch failed ({source['name']}): {e}")
        return []

def fetch_github_trending(source: dict) -> list[dict]:
    """Fetch trending AI/LLM repos from GitHub (daily)."""
    try:
        # Using a free, reliable mirror for GitHub trending
        url = "https://gtrend.yapie.me/repositories?since=daily&language=python"
        r = requests.get(url, timeout=10)
        data = r.json()
        items = []
        for repo in data[:8]:
            name = repo.get('name', '')
            description = repo.get('description', '') or "No description"
            items.append({
                'title': f"⭐ {repo.get('stars', 0)} stars · {name}",
                'summary': description[:400],
                'url': repo.get('url', '#'),
                'source': 'GitHub Trending (AI/LLM)',
                'priority': source['priority'],
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

def fetch_all_news() -> list[dict]:
    log.info("Fetching news from all sources...")
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
        # Mark as seen now (will be saved at end)
        seen[h] = now
        unique_items.append(item)

    # Diversity: max 2 items from same source
    source_count = {}
    diverse_items = []
    for item in unique_items:
        src = item['source']
        if source_count.get(src, 0) < 2:
            diverse_items.append(item)
            source_count[src] = source_count.get(src, 0) + 1
        if len(diverse_items) >= MAX_ITEMS:
            break

    top = diverse_items[:MAX_ITEMS]

    # Persist seen titles (auto-expiry on next run)
    save_seen_titles(seen)

    log.info(f"Selected {len(top)} fresh items after scoring, dedup, and diversity")
    return top

# ============================================================================
# (Everything below this line is unchanged from original)
# - write_digest_with_llm
# - fetch_my_blogs
# - fetch_subscribers
# - build_email_html
# - send_newsletter
# - main
# ============================================================================

def write_digest_with_llm(news_items: list[dict], date_str: str) -> dict:
    news_text = '\n\n'.join([
        f"[{i+1}] SOURCE: {item['source']}\nTITLE: {item['title']}\nSUMMARY: {item['summary']}\nURL: {item['url']}"
        for i, item in enumerate(news_items)
    ])

    system_prompt = """You are writing a daily AI newsletter for Dr. Prateek Singh,
Senior Manager of GenAI at Samsung Research Institute, Noida.
IIT Roorkee PhD. Expert in LLM deployment, on-device AI, quantization,
AI agents, and edge inference.

VOICE: Confident, clear, slightly technical but accessible. Not hype-y.
Write like a senior engineer who has seen a lot of AI trends come and go.
Short sentences. No fluff. Respect the reader's time.

FOCUS: LLMs, AI agents, model releases, inference optimization, open-source AI,
on-device AI, quantization, agent frameworks, reasoning models.

OUTPUT FORMAT — return valid JSON only, no markdown, no explanation:
{
  "subject": "email subject line (max 70 chars, include date and 1-2 key topics)",
  "top_story": {
    "headline": "one punchy line",
    "body": "2-3 sentences explaining why it matters. Plain English.",
    "url": "url from the news items"
  },
  "llm_spotlight": {
    "headline": "one punchy line about a new model, agent framework, or inference breakthrough",
    "body": "2-3 sentences — what changed, what it means for practitioners",
    "url": "url"
  },
  "papers": [
    {"title": "short title", "summary": "one sentence — what it does and why it matters", "url": "url"},
    {"title": "short title", "summary": "one sentence", "url": "url"},
    {"title": "short title", "summary": "one sentence", "url": "url"}
  ],
  "tools_repos": {
    "headline": "tool or repo name",
    "body": "1-2 sentences on what it does and why it matters for LLM/agent practitioners",
    "url": "url"
  },
  "closing_thought": "1 short sentence — an honest observation or provocative question about today's AI landscape. No positivity fluff."
}"""

    user_prompt = f"""Date: {date_str}

Here are today's top AI/LLM news items — write the newsletter digest:

{news_text}

Return only the JSON object. No markdown. No explanation."""

    log.info("Calling Groq API...")
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }

    models_to_try = [
        ('llama3-70b-8192',          True),
        ('llama-3.3-70b-versatile',  True),
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
                'max_tokens': 1200,
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
            r.raise_for_status()
            content = r.json()['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()
            digest = json.loads(content)
            log.info(f"Digest written with {model_name}. Subject: {digest.get('subject', 'N/A')}")
            return digest
        except Exception as e:
            log.warning(f"  Model {model_name} failed: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

def fetch_my_blogs(max_posts: int = 3) -> list[dict]:
    try:
        from html.parser import HTMLParser
        r = requests.get('https://prateeksinghphd.in/blogs.html', timeout=10)
        r.raise_for_status()
        html = r.text

        class BlogParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.posts = []
                self.in_card = False
                self.in_title = False
                self.in_date = False
                self.in_excerpt = False
                self.current = {}
                self.depth = 0
                self.card_depth = 0

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                cls = attrs_dict.get('class', '')
                self.depth += 1
                if tag in ('a', 'div') and any(x in cls for x in ['post-card', 'blog-card', 'blog-post']):
                    self.in_card = True
                    self.card_depth = self.depth
                    self.current = {
                        'url': attrs_dict.get('href', '#'),
                        'title': '', 'date': '', 'excerpt': ''
                    }
                    if not self.current['url'].startswith('http'):
                        self.current['url'] = 'https://prateeksinghphd.in/' + self.current['url'].lstrip('/')
                if self.in_card:
                    if any(x in cls for x in ['post-title', 'blog-card-title']):
                        self.in_title = True
                    if any(x in cls for x in ['post-date', 'blog-card-date']):
                        self.in_date = True
                    if any(x in cls for x in ['post-excerpt', 'blog-card-excerpt']):
                        self.in_excerpt = True

            def handle_endtag(self, tag):
                self.depth -= 1
                if self.in_card and self.depth < self.card_depth:
                    if self.current.get('title'):
                        self.posts.append(self.current.copy())
                    self.in_card = False
                    self.current = {}
                self.in_title = False
                self.in_date = False
                self.in_excerpt = False

            def handle_data(self, data):
                data = data.strip()
                if not data or not self.in_card:
                    return
                if self.in_title and not self.current.get('title'):
                    self.current['title'] = data
                elif self.in_date and not self.current.get('date'):
                    self.current['date'] = data
                elif self.in_excerpt and not self.current.get('excerpt'):
                    self.current['excerpt'] = data

        parser = BlogParser()
        parser.feed(html)
        posts = parser.posts[:max_posts]
        if not posts:
            log.warning("Blog parser got no results — using fallback list")
            posts = _blog_fallback()
        log.info(f"Fetched {len(posts)} blog posts")
        return posts
    except Exception as e:
        log.warning(f"Blog fetch failed: {e} — using fallback")
        return _blog_fallback()

def _blog_fallback() -> list[dict]:
    return [
        {
            'title': 'The Agent Wars: OpenClaw, NemoClaw & Hermes',
            'date': 'Apr 04, 2026',
            'excerpt': 'OpenClaw became the OS for personal AI. NemoClaw made it enterprise-safe. Hermes made it evolve.',
            'url': 'https://prateeksinghphd.in/agentic.html'
        },
        {
            'title': 'TurboQuant',
            'date': 'Mar 29, 2026',
            'excerpt': 'Google solved the KV cache bottleneck — 6× compression, 8× speedup, zero accuracy loss.',
            'url': 'https://prateeksinghphd.in/turboquant.html'
        },
        {
            'title': 'Quantization in LLMs',
            'date': 'Mar 21, 2026',
            'excerpt': 'Making 70B models fit in 24 GB without making them dumber — GPTQ, AWQ, NF4 and beyond.',
            'url': 'https://prateeksinghphd.in/quantization-llms.html'
        },
    ]

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

def build_email_html(digest: dict, date_str: str, email: str, blogs: list = None) -> str:
    unsubscribe_url = f"https://prateeksinghphd.in/api/unsubscribe?email={requests.utils.quote(email)}"
    paper_colors = ['#00d9b4', '#7c6bff', '#ff6b9d']
    papers_html = ''
    for i, p in enumerate(digest.get('papers', [])[:3]):
        color = paper_colors[i % len(paper_colors)]
        papers_html += f"""
        <div style="margin-bottom:20px;background:#0d0d1a;border:1px solid #1e1e35;
                    border-left:3px solid {color};border-radius:6px;padding:20px 24px;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <span style="font-family:Georgia,serif;font-size:28px;font-weight:900;
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
    top = digest.get('top_story', {})
    llm = digest.get('llm_spotlight', {})
    tools = digest.get('tools_repos', {})
    closing = digest.get('closing_thought', '')
    subject = digest.get('subject', f'AI Daily — {date_str}')
    blogs = blogs or []
    blog_rows_html = ''
    for i, b in enumerate(blogs[:3]):
        border_top = 'border-top:1px solid #1e1e35;' if i > 0 else ''
        blog_rows_html += f"""
        <div style="padding:16px 0;{border_top}display:flex;gap:16px;align-items:flex-start;">
          <div style="background:#00d9b414;border:1px solid #00d9b430;border-radius:4px;
                      padding:6px 10px;flex-shrink:0;text-align:center;min-width:44px;">
            <span style="font-family:Georgia,serif;font-size:18px;font-weight:900;
                         color:#00d9b4;line-height:1;">✍️</span>
          </div>
          <div style="flex:1;">
            <a href="{b.get('url','https://prateeksinghphd.in/blogs.html')}"
               style="font-size:16px;font-weight:700;color:#f0f0f8;
                      text-decoration:none;line-height:1.3;display:block;margin-bottom:5px;">
              {b.get('title','')}
            </a>
            <p style="font-size:13px;color:#6a6a8a;margin:0 0 8px;font-family:monospace;">
              {b.get('date','')}
            </p>
            <p style="font-size:14px;color:#9090b8;margin:0 0 10px;line-height:1.6;">
              {b.get('excerpt','')}
            </p>
            <a href="{b.get('url','https://prateeksinghphd.in/blogs.html')}"
               style="font-family:monospace;font-size:11px;letter-spacing:2px;
                      color:#00d9b4;text-transform:uppercase;text-decoration:none;">
              Read →
            </a>
          </div>
        </div>"""
    blogs_section_html = ''
    if blog_rows_html:
        blogs_section_html = f"""
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;background:#0a0a14;">
    <div style="display:inline-block;background:#00d9b414;border:1px solid #00d9b430;
                border-radius:3px;padding:4px 12px;margin-bottom:20px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;
                   color:#00d9b4;text-transform:uppercase;font-weight:700;">
        ✍️ From My Blog
      </span>
    </div>
    <p style="font-size:14px;color:#5a5a7a;margin:0 0 20px;">
      Recent posts from <a href="https://prateeksinghphd.in/blogs.html"
      style="color:#00d9b4;text-decoration:none;">prateeksinghphd.in</a>
    </p>
    {blog_rows_html}
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid #1e1e35;">
      <a href="https://prateeksinghphd.in/blogs.html"
         style="display:inline-block;background:#00d9b414;color:#00d9b4;
                border:1px solid #00d9b430;font-family:monospace;font-size:11px;
                letter-spacing:2px;font-weight:700;text-transform:uppercase;
                text-decoration:none;padding:10px 20px;border-radius:3px;">
        View All Blogs →
      </a>
    </div>
  </div>"""
    try:
        dow = datetime.now().strftime('%A').upper()
    except Exception:
        dow = 'TODAY'
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>{subject}</title></head>
<body style="margin:0;padding:0;background:#08080f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
<div style="max-width:620px;margin:0 auto;background:#08080f;">
  <div style="background:#00d9b4;padding:10px 40px;text-align:center;">
    <span style="font-family:monospace;font-size:11px;font-weight:700;letter-spacing:3px;color:#08080f;text-transform:uppercase;">
      🧠 AI DAILY BRIEFING &nbsp;·&nbsp; {dow} &nbsp;·&nbsp; 6 PM IST
    </span>
  </div>
  <div style="padding:40px 44px 32px;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr><td><p style="font-family:Georgia,serif;font-size:13px;font-style:italic;color:#00d9b4;margin:0 0 6px;">Dr. Prateek Singh</p>
          <h1 style="font-family:Georgia,serif;font-size:36px;font-weight:900;color:#f0f0f8;margin:0;line-height:1.1;letter-spacing:-0.5px;">Your AI Briefing</h1>
          <p style="font-size:15px;color:#5a5a7a;margin:8px 0 0;font-family:monospace;">{date_str}</p></td>
        <td style="text-align:right;vertical-align:top;padding-top:4px;"><div style="background:#0d0d1a;border:1px solid #1e1e35;border-radius:6px;padding:10px 16px;display:inline-block;">
          <p style="font-family:monospace;font-size:10px;letter-spacing:2px;color:#5a5a7a;text-transform:uppercase;margin:0 0 3px;">Today</p>
          <p style="font-family:Georgia,serif;font-size:22px;font-weight:900;color:#00d9b4;margin:0;line-height:1;">5 min</p>
          <p style="font-family:monospace;font-size:9px;color:#3a3a5a;margin:2px 0 0;text-transform:uppercase;">read</p>
        </div></td></tr>
    </table>
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;">
    <div style="display:inline-block;background:#00d9b414;border:1px solid #00d9b430;border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#00d9b4;text-transform:uppercase;font-weight:700;">① Top Story</span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:26px;font-weight:900;color:#f0f0f8;margin:0 0 14px;line-height:1.25;">
      <a href="{top.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{top.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">{top.get('body','')}</p>
    <a href="{top.get('url','#')}" style="display:inline-block;background:#00d9b4;color:#08080f;font-family:monospace;font-size:12px;letter-spacing:2px;font-weight:700;text-transform:uppercase;text-decoration:none;padding:12px 24px;border-radius:3px;">Read Full Story →</a>
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;background:#0c0c18;">
    <div style="display:inline-block;background:#7c6bff14;border:1px solid #7c6bff30;border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#7c6bff;text-transform:uppercase;font-weight:700;">🤖 LLM &amp; Agents Spotlight</span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:24px;font-weight:900;color:#f0f0f8;margin:0 0 14px;line-height:1.3;">
      <a href="{llm.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{llm.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">{llm.get('body','')}</p>
    <a href="{llm.get('url','#')}" style="display:inline-block;background:#7c6bff18;color:#7c6bff;border:1px solid #7c6bff40;font-family:monospace;font-size:12px;letter-spacing:2px;font-weight:700;text-transform:uppercase;text-decoration:none;padding:12px 24px;border-radius:3px;">Dig Deeper →</a>
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;">
    <div style="display:inline-block;background:#7c6bff14;border:1px solid #7c6bff30;border-radius:3px;padding:4px 12px;margin-bottom:22px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#7c6bff;text-transform:uppercase;font-weight:700;">📄 3 Papers Worth Reading</span>
    </div>
    {papers_html}
  </div>
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;background:#0c0c18;">
    <div style="display:inline-block;background:#ffb34714;border:1px solid #ffb34730;border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#ffb347;text-transform:uppercase;font-weight:700;">🚀 Tool / Repo of the Day</span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:22px;font-weight:900;color:#f0f0f8;margin:0 0 12px;line-height:1.3;">
      <a href="{tools.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{tools.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">{tools.get('body','')}</p>
    <a href="{tools.get('url','#')}" style="display:inline-block;background:#ffb34718;color:#ffb347;border:1px solid #ffb34740;font-family:monospace;font-size:12px;letter-spacing:2px;font-weight:700;text-transform:uppercase;text-decoration:none;padding:12px 24px;border-radius:3px;">Check it out →</a>
  </div>
  <div style="padding:30px 44px;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr><td style="width:4px;background:linear-gradient(180deg,#00d9b4,#7c6bff);border-radius:2px;">&nbsp;</td>
    <td style="padding-left:20px;"><p style="font-family:Georgia,serif;font-size:17px;font-style:italic;color:#8a8aaa;line-height:1.75;margin:0;">"{closing}"</p>
    <p style="font-size:13px;color:#4a4a6a;margin:10px 0 0;font-family:monospace;">— Dr. Prateek Singh</p></td></tr></table>
  </div>
  {blogs_section_html}
  <div style="padding:32px 44px;background:#0d0d1a;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr><td style="vertical-align:middle;">
      <p style="font-family:Georgia,serif;font-size:18px;font-weight:700;color:#f0f0f8;margin:0 0 4px;">Building with LLMs or AI Agents?</p>
      <p style="font-size:14px;color:#6a6a8a;margin:0;">Let's discuss your project — free 30-min call.</p></td>
    <td style="text-align:right;vertical-align:middle;"><a href="https://cal.com/prateek-singh-la8jpj" style="display:inline-block;background:#00d9b4;color:#08080f;font-family:monospace;font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase;text-decoration:none;padding:12px 20px;border-radius:3px;white-space:nowrap;">Book a Call →</a></td></tr></table>
  </div>
  <div style="padding:32px 44px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;"><tr><td>
      <p style="font-family:monospace;font-size:10px;letter-spacing:2px;color:#00d9b4;text-transform:uppercase;margin:0 0 8px;">Dr. Prateek Singh</p>
      <p style="font-size:13px;color:#4a4a6a;line-height:1.8;margin:0;">Senior Manager, GenAI &amp; Digital Health AI<br>Samsung Research Institute, Noida · IIT Roorkee PhD</p></td>
    <td style="text-align:right;vertical-align:middle;"><a href="https://prateeksinghphd.in" style="display:inline-block;background:#0d0d1a;color:#00d9b4;border:1px solid #1e1e35;font-family:monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;text-decoration:none;padding:8px 14px;border-radius:3px;">My Blog →</a></td></tr></table>
    <div style="border-top:1px solid #1e1e35;padding-top:20px;">
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

def send_newsletter(emails: list[str], digest: dict, date_str: str, blogs: list = None) -> dict:
    results = {'sent': 0, 'failed': 0, 'errors': []}
    log.info(f"Sending to {len(emails)} subscribers...")
    for i, email in enumerate(emails):
        try:
            html = build_email_html(digest, date_str, email, blogs=blogs)
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
    log.info("Recommended schedule: 6 PM IST (12:30 UTC) to capture full day's news")
    log.info(f"{'='*60}")

    news_items = fetch_all_news()
    if not news_items:
        log.error("No news items fetched — aborting")
        return

    my_blogs = fetch_my_blogs(max_posts=3)
    digest = write_digest_with_llm(news_items, date_str)
    emails = fetch_subscribers()

    if not emails:
        log.error("No subscribers found — aborting")
        return

    results = send_newsletter(emails, digest, date_str, blogs=my_blogs)

    log.info(f"{'='*60}")
    log.info(f"✅ Sent:   {results['sent']}")
    log.info(f"❌ Failed: {results['failed']}")
    if results['errors']:
        log.warning(f"Errors: {json.dumps(results['errors'], indent=2)}")
    log.info(f"{'='*60}")

if __name__ == '__main__':
    main()
