"""
╔══════════════════════════════════════════════════════════════════╗
║  Daily AI Newsletter Agent                                       ║
║  Dr. Prateek Singh · prateeksinghphd.in                         ║
║                                                                  ║
║  What this does:                                                 ║
║  1. Fetches latest AI news from ArXiv, HuggingFace, GitHub,     ║
║     Hacker News, and RSS feeds                                   ║
║  2. Filters for Healthcare AI, LLMs, Edge AI relevance          ║
║  3. Uses Hermes-3 (via Groq) to write digest in your voice      ║
║  4. Fetches subscriber list from Cloudflare Worker               ║
║  5. Sends personalised email via Resend at 9 AM IST             ║
║                                                                  ║
║  Run manually:  python agent.py                                  ║
║  Scheduled:     GitHub Actions cron (see .github/workflows/)     ║
╚══════════════════════════════════════════════════════════════════╝
 
INSTALL DEPENDENCIES:
    pip install requests feedparser groq resend python-dotenv
 
ENVIRONMENT VARIABLES (.env file or GitHub Secrets):
    GROQ_API_KEY       = gsk_xxxx...      (groq.com — free, Hermes-3 available)
    RESEND_API_KEY     = re_xxxx...       (resend.com — free 3K/month)
    ADMIN_TOKEN        = your-admin-token (same as in Cloudflare Worker)
    SUBSCRIBERS_URL    = https://prateeksinghphd.in/api/subscribers
    FROM_EMAIL         = newsletter@prateeksinghphd.in
    FROM_NAME          = Dr. Prateek Singh
"""
 
import os
import json
import time
import logging
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
MODEL           = 'llama3-70b-8192'  # Groq — stable, supports json_object
MAX_ITEMS       = 6   # max news items to pass to the LLM
TEST_MODE       = os.environ.get('TEST_MODE', 'false').lower() == 'true'
TEST_EMAIL      = os.environ.get('TEST_EMAIL', 'prateek29singh@gmail.com')
 
# ── RSS / API sources
SOURCES = [
    # ArXiv — Healthcare AI + ML
    {
        'name': 'ArXiv CS.AI',
        'url': 'https://rss.arxiv.org/rss/cs.AI',
        'type': 'rss',
        'priority': 1
    },
    {
        'name': 'ArXiv eess.SP (Signal Processing)',
        'url': 'https://rss.arxiv.org/rss/eess.SP',
        'type': 'rss',
        'priority': 1
    },
    {
        'name': 'ArXiv cs.LG (Machine Learning)',
        'url': 'https://rss.arxiv.org/rss/cs.LG',
        'type': 'rss',
        'priority': 1
    },
    # HuggingFace daily papers
    {
        'name': 'HuggingFace Daily Papers',
        'url': 'https://huggingface.co/papers',
        'type': 'hf_papers',
        'priority': 2
    },
    # Hacker News — AI stories
    {
        'name': 'Hacker News',
        'url': 'https://hn.algolia.com/api/v1/search?tags=story&query=AI+machine+learning&hitsPerPage=10',
        'type': 'hn_api',
        'priority': 2
    },
    # AI-focused RSS feeds
    {
        'name': 'MIT Tech Review AI',
        'url': 'https://www.technologyreview.com/feed/',
        'type': 'rss',
        'priority': 3
    },
    {
        'name': 'Google AI Blog',
        'url': 'https://blog.research.google/feeds/posts/default?alt=rss',
        'type': 'rss',
        'priority': 2
    },
    {
        'name': 'DeepMind Blog',
        'url': 'https://deepmind.google/blog/rss.xml',
        'type': 'rss',
        'priority': 2
    },
]
 
# ── Keywords for relevance filtering
HIGH_PRIORITY_KEYWORDS = [
    'ecg', 'ppg', 'blood pressure', 'arrhythmia', 'wearable',
    'healthcare ai', 'clinical ai', 'medical ai', 'health ai',
    'on-device', 'edge ai', 'edge inference', 'mobile ai',
    'quantization', 'llm inference', 'gguf', 'llama', 'mistral',
    'mamba', 'ssm', 'npu', 'qualcomm', 'samsung ai',
]
GENERAL_AI_KEYWORDS = [
    'large language model', 'llm', 'transformer', 'attention',
    'fine-tuning', 'rlhf', 'rag', 'retrieval', 'embedding',
    'diffusion', 'multimodal', 'foundation model', 'ai agent',
    'neural network', 'deep learning', 'machine learning',
]
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 1: FETCH NEWS
# ══════════════════════════════════════════════════════════════════
 
def fetch_rss(source: dict) -> list[dict]:
    """Fetch and parse an RSS feed."""
    try:
        feed = feedparser.parse(source['url'])
        items = []
        for entry in feed.entries[:8]:
            items.append({
                'title':   entry.get('title', '').strip(),
                'summary': entry.get('summary', entry.get('description', ''))[:400].strip(),
                'url':     entry.get('link', ''),
                'source':  source['name'],
                'priority': source['priority']
            })
        log.info(f"  {source['name']}: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  RSS fetch failed for {source['name']}: {e}")
        return []
 
 
def fetch_hn(source: dict) -> list[dict]:
    """Fetch AI stories from Hacker News Algolia API."""
    try:
        r = requests.get(source['url'], timeout=10)
        data = r.json()
        items = []
        for hit in data.get('hits', []):
            items.append({
                'title':   hit.get('title', '').strip(),
                'summary': f"Points: {hit.get('points', 0)} | Comments: {hit.get('num_comments', 0)}",
                'url':     hit.get('url') or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                'source':  'Hacker News',
                'priority': source['priority']
            })
        log.info(f"  Hacker News: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  HN fetch failed: {e}")
        return []
 
 
def fetch_hf_papers(source: dict) -> list[dict]:
    """Fetch latest papers from HuggingFace daily papers API."""
    try:
        r = requests.get('https://huggingface.co/api/daily_papers', timeout=10)
        papers = r.json()
        items = []
        for p in papers[:6]:
            paper = p.get('paper', {})
            items.append({
                'title':   paper.get('title', '').strip(),
                'summary': paper.get('summary', '')[:400].strip(),
                'url':     f"https://huggingface.co/papers/{paper.get('id', '')}",
                'source':  'HuggingFace Papers',
                'priority': source['priority']
            })
        log.info(f"  HuggingFace Papers: {len(items)} items")
        return items
    except Exception as e:
        log.warning(f"  HF papers fetch failed: {e}")
        return []
 
 
def score_item(item: dict) -> float:
    """Score news item by relevance to our topics."""
    text = (item['title'] + ' ' + item['summary']).lower()
    score = 0.0
 
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in text:
            score += 3.0
 
    for kw in GENERAL_AI_KEYWORDS:
        if kw in text:
            score += 1.0
 
    # Penalise by source priority (lower number = higher priority)
    score -= (item['priority'] - 1) * 0.5
 
    return score
 
 
def fetch_all_news() -> list[dict]:
    """Fetch from all sources, score, deduplicate, and return top items."""
    log.info("Fetching news from all sources...")
    all_items = []
 
    for source in SOURCES:
        if source['type'] == 'rss':
            all_items.extend(fetch_rss(source))
        elif source['type'] == 'hn_api':
            all_items.extend(fetch_hn(source))
        elif source['type'] == 'hf_papers':
            all_items.extend(fetch_hf_papers(source))
        time.sleep(0.5)  # be polite to servers
 
    # Score and sort
    for item in all_items:
        item['score'] = score_item(item)
 
    # Deduplicate by title similarity (simple)
    seen_titles = set()
    unique_items = []
    for item in sorted(all_items, key=lambda x: x['score'], reverse=True):
        title_key = item['title'].lower()[:40]
        if title_key not in seen_titles and item['title']:
            seen_titles.add(title_key)
            unique_items.append(item)
 
    top = unique_items[:MAX_ITEMS]
    log.info(f"Selected {len(top)} items after scoring and deduplication")
    return top
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 2: WRITE DIGEST WITH HERMES (via Groq)
# ══════════════════════════════════════════════════════════════════
 
def write_digest_with_llm(news_items: list[dict], date_str: str) -> dict:
    """Use Hermes-3 on Groq to write the newsletter digest."""
 
    news_text = '\n\n'.join([
        f"[{i+1}] SOURCE: {item['source']}\nTITLE: {item['title']}\nSUMMARY: {item['summary']}\nURL: {item['url']}"
        for i, item in enumerate(news_items)
    ])
 
    system_prompt = """You are writing a daily AI newsletter for Dr. Prateek Singh, 
Senior Manager of GenAI & Digital Health AI at Samsung Research Institute, Noida. 
IIT Roorkee PhD. Expert in Healthcare AI, ECG/PPG signal processing, LLM deployment, 
and on-device AI.
 
VOICE: Confident, clear, slightly technical but accessible. Not hype-y. 
Write like a senior engineer who has seen a lot of AI trends come and go.
Short sentences. No fluff. Respect the reader's time.
 
OUTPUT FORMAT — return valid JSON only, no markdown, no explanation:
{
  "subject": "email subject line (max 70 chars, include date and 1-2 key topics)",
  "top_story": {
    "headline": "one punchy line",
    "body": "2-3 sentences explaining why it matters. Plain English.",
    "url": "url from the news items"
  },
  "healthcare_spotlight": {
    "headline": "one punchy line (if no healthcare item, pick the most relevant)",
    "body": "2-3 sentences",
    "url": "url"
  },
  "papers": [
    {"title": "short title", "summary": "one sentence", "url": "url"},
    {"title": "short title", "summary": "one sentence", "url": "url"},
    {"title": "short title", "summary": "one sentence", "url": "url"}
  ],
  "tools_repos": {
    "headline": "tool or repo name",
    "body": "1-2 sentences on what it does and why it matters",
    "url": "url"
  },
  "closing_thought": "1 short sentence — an honest observation or provocative question about today's AI landscape. No positivity fluff."
}"""
 
    user_prompt = f"""Date: {date_str}
 
Here are today's top AI news items — write the newsletter digest:
 
{news_text}
 
Return only the JSON object. No markdown. No explanation."""
 
    log.info("Calling Groq API...")
 
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
 
    # Try primary model first, fall back to mixtral if it fails
    models_to_try = [
        ('llama3-70b-8192',          True),   # (model_name, supports_json_mode)
        ('llama-3.3-70b-versatile',  True),
        ('mixtral-8x7b-32768',       False),  # mixtral doesn't support json_object
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
 
            content = r.json()['choices'][0]['message']['content']
 
            # Robust JSON extraction — strip markdown fences if present
            content = content.strip()
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
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 2.5: FETCH MY LATEST BLOG POSTS
# ══════════════════════════════════════════════════════════════════
 
def fetch_my_blogs(max_posts: int = 3) -> list[dict]:
    """Scrape latest blog posts from prateeksinghphd.in/blogs.html"""
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
 
                # Detect post card
                if tag in ('a', 'div') and any(x in cls for x in ['post-card', 'blog-card', 'blog-post']):
                    self.in_card = True
                    self.card_depth = self.depth
                    self.current = {
                        'url': attrs_dict.get('href', '#'),
                        'title': '',
                        'date': '',
                        'excerpt': ''
                    }
                    # Make URL absolute
                    if self.current['url'].startswith('/') or not self.current['url'].startswith('http'):
                        if not self.current['url'].startswith('http'):
                            self.current['url'] = 'https://prateeksinghphd.in/' + self.current['url'].lstrip('/')
 
                if self.in_card:
                    if any(x in cls for x in ['post-title', 'blog-card-title', 'post-card-title']):
                        self.in_title = True
                    if any(x in cls for x in ['post-date', 'blog-card-date', 'post-card-date']):
                        self.in_date = True
                    if any(x in cls for x in ['post-excerpt', 'blog-card-excerpt', 'post-card-excerpt']):
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
 
        # Fallback — if parser gets nothing, use hardcoded latest posts
        if not posts:
            log.warning("Blog parser got no results — using fallback list")
            posts = [
                {
                    'title': 'TurboQuant',
                    'date': 'Mar 29, 2026',
                    'excerpt': 'Google solved the KV cache bottleneck — 6× compression, 8× speedup, zero accuracy loss.',
                    'url': 'https://prateeksinghphd.in/turboquant.html'
                },
                {
                    'title': 'LLM Inference Runtimes',
                    'date': 'Mar 22, 2026',
                    'excerpt': 'GGUF, TensorRT, QNN — a deep dive into every major runtime across all hardware.',
                    'url': 'https://prateeksinghphd.in/inference-runtimes.html'
                },
                {
                    'title': 'Quantization in LLMs',
                    'date': 'Mar 21, 2026',
                    'excerpt': 'Making 70B models fit in 24 GB without making them dumber — GPTQ, AWQ, NF4 and beyond.',
                    'url': 'https://prateeksinghphd.in/quantization-llms.html'
                },
            ]
 
        log.info(f"Fetched {len(posts)} blog posts from prateeksinghphd.in")
        for p in posts:
            log.info(f"  Blog: {p.get('title','?')} ({p.get('date','?')})")
        return posts
 
    except Exception as e:
        log.warning(f"Blog fetch failed: {e} — using fallback")
        return [
            {
                'title': 'TurboQuant',
                'date': 'Mar 29, 2026',
                'excerpt': 'Google solved the KV cache bottleneck — 6× compression, 8× speedup, zero accuracy loss.',
                'url': 'https://prateeksinghphd.in/turboquant.html'
            },
            {
                'title': 'Quantization in LLMs',
                'date': 'Mar 21, 2026',
                'excerpt': 'Making 70B models fit in 24 GB without making them dumber.',
                'url': 'https://prateeksinghphd.in/quantization-llms.html'
            },
        ]
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 3: FETCH SUBSCRIBERS
# ══════════════════════════════════════════════════════════════════
 
def fetch_subscribers() -> list[str]:
    """Fetch active subscriber emails from Cloudflare Worker."""
    if TEST_MODE:
        log.info(f"TEST MODE — using test email: {TEST_EMAIL}")
        return [TEST_EMAIL]
 
    log.info("Fetching subscribers from Cloudflare...")
    try:
        # If token already in URL (e.g. ?token=xxx), use as-is
        # Otherwise append it
        if 'token=' in SUBSCRIBERS_URL or 'Authorization' in SUBSCRIBERS_URL:
            url = SUBSCRIBERS_URL
        else:
            url = f"{SUBSCRIBERS_URL}?token={ADMIN_TOKEN}"
 
        log.info(f"Fetching from: {url.split('?')[0]}...")  # log URL without token
 
        r = requests.get(
            url,
            headers={'Authorization': f'Bearer {ADMIN_TOKEN}'},
            timeout=15
        )
 
        log.info(f"Subscriber API status: {r.status_code}")
        r.raise_for_status()
        data = r.json()
 
        log.info(f"API response keys: {list(data.keys())}")
        log.info(f"Total in response: {data.get('count', 'N/A')}")
 
        all_subs = data.get('subscribers', [])
        log.info(f"Raw records: {len(all_subs)}")
 
        for s in all_subs[:3]:
            log.info(f"  Sample: email={s.get('email')} active={s.get('active')}")
 
        emails = [
            s['email'] for s in all_subs
            if s.get('active', True) is not False
            and s.get('email')
        ]
 
        log.info(f"Active subscribers to send: {len(emails)}")
        log.info(f"Email list: {emails}")
        return emails
 
    except Exception as e:
        log.error(f"Failed to fetch subscribers: {e}")
        raise
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 4: BUILD HTML EMAIL
# ══════════════════════════════════════════════════════════════════
 
def build_email_html(digest: dict, date_str: str, email: str, blogs: list = None) -> str:
    """Build premium HTML email from the digest."""
 
    unsubscribe_url = f"https://prateeksinghphd.in/api/unsubscribe?email={requests.utils.quote(email)}"
 
    # Paper cards — numbered, with full link button
    paper_colors = ['#00d9b4', '#7c6bff', '#ff6b9d']
    papers_html  = ''
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
 
    top     = digest.get('top_story', {})
    health  = digest.get('healthcare_spotlight', {})
    tools   = digest.get('tools_repos', {})
    closing = digest.get('closing_thought', '')
    subject = digest.get('subject', f'AI Daily — {date_str}')
 
    # Build blog rows HTML
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
  <!-- ══ FROM MY BLOG ══ -->
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
 
    # Day of week for the header
    try:
        from datetime import datetime as dt
        dow = dt.now().strftime('%A').upper()
    except Exception:
        dow = 'TODAY'
 
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#08080f;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
 
<div style="max-width:620px;margin:0 auto;background:#08080f;">
 
  <!-- ══ TOP BAR ══ -->
  <div style="background:#00d9b4;padding:10px 40px;text-align:center;">
    <span style="font-family:monospace;font-size:11px;font-weight:700;
                 letter-spacing:3px;color:#08080f;text-transform:uppercase;">
      🧠 AI DAILY BRIEFING &nbsp;·&nbsp; {dow} &nbsp;·&nbsp; 9 AM IST
    </span>
  </div>
 
  <!-- ══ HEADER ══ -->
  <div style="padding:40px 44px 32px;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td>
          <p style="font-family:Georgia,serif;font-size:13px;font-style:italic;
                    color:#00d9b4;margin:0 0 6px;">Dr. Prateek Singh</p>
          <h1 style="font-family:Georgia,serif;font-size:36px;font-weight:900;
                     color:#f0f0f8;margin:0;line-height:1.1;letter-spacing:-0.5px;">
            Your AI Briefing
          </h1>
          <p style="font-size:15px;color:#5a5a7a;margin:8px 0 0;font-family:monospace;">
            {date_str}
          </p>
        </td>
        <td style="text-align:right;vertical-align:top;padding-top:4px;">
          <div style="background:#0d0d1a;border:1px solid #1e1e35;border-radius:6px;
                      padding:10px 16px;display:inline-block;">
            <p style="font-family:monospace;font-size:10px;letter-spacing:2px;
                      color:#5a5a7a;text-transform:uppercase;margin:0 0 3px;">Today</p>
            <p style="font-family:Georgia,serif;font-size:22px;font-weight:900;
                      color:#00d9b4;margin:0;line-height:1;">5 min</p>
            <p style="font-family:monospace;font-size:9px;color:#3a3a5a;
                      margin:2px 0 0;text-transform:uppercase;">read</p>
          </div>
        </td>
      </tr>
    </table>
  </div>
 
  <!-- ══ TOP STORY ══ -->
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;">
    <div style="display:inline-block;background:#00d9b414;border:1px solid #00d9b430;
                border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;
                   color:#00d9b4;text-transform:uppercase;font-weight:700;">
        ① Top Story
      </span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:26px;font-weight:900;
               color:#f0f0f8;margin:0 0 14px;line-height:1.25;">
      <a href="{top.get('url','#')}"
         style="color:#f0f0f8;text-decoration:none;">{top.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">
      {top.get('body','')}
    </p>
    <a href="{top.get('url','#')}"
       style="display:inline-block;background:#00d9b4;color:#08080f;font-family:monospace;
              font-size:12px;letter-spacing:2px;font-weight:700;text-transform:uppercase;
              text-decoration:none;padding:12px 24px;border-radius:3px;">
      Read Full Story →
    </a>
  </div>
 
  <!-- ══ HEALTHCARE SPOTLIGHT ══ -->
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;background:#0c0c18;">
    <div style="display:inline-block;background:#ff6b9d14;border:1px solid #ff6b9d30;
                border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;
                   color:#ff6b9d;text-transform:uppercase;font-weight:700;">
        ❤️ Healthcare AI Spotlight
      </span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:24px;font-weight:900;
               color:#f0f0f8;margin:0 0 14px;line-height:1.3;">
      <a href="{health.get('url','#')}"
         style="color:#f0f0f8;text-decoration:none;">{health.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">
      {health.get('body','')}
    </p>
    <a href="{health.get('url','#')}"
       style="display:inline-block;background:#ff6b9d18;color:#ff6b9d;
              border:1px solid #ff6b9d40;font-family:monospace;font-size:12px;
              letter-spacing:2px;font-weight:700;text-transform:uppercase;
              text-decoration:none;padding:12px 24px;border-radius:3px;">
      Read More →
    </a>
  </div>
 
  <!-- ══ PAPERS ══ -->
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;">
    <div style="display:inline-block;background:#7c6bff14;border:1px solid #7c6bff30;
                border-radius:3px;padding:4px 12px;margin-bottom:22px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;
                   color:#7c6bff;text-transform:uppercase;font-weight:700;">
        📄 3 Papers Worth Reading
      </span>
    </div>
    {papers_html}
  </div>
 
  <!-- ══ TOOL OF THE DAY ══ -->
  <div style="padding:36px 44px;border-bottom:1px solid #1e1e35;background:#0c0c18;">
    <div style="display:inline-block;background:#ffb34714;border:1px solid #ffb34730;
                border-radius:3px;padding:4px 12px;margin-bottom:18px;">
      <span style="font-family:monospace;font-size:10px;letter-spacing:3px;
                   color:#ffb347;text-transform:uppercase;font-weight:700;">
        🚀 Tool / Repo of the Day
      </span>
    </div>
    <h2 style="font-family:Georgia,serif;font-size:22px;font-weight:900;
               color:#f0f0f8;margin:0 0 12px;line-height:1.3;">
      <a href="{tools.get('url','#')}"
         style="color:#f0f0f8;text-decoration:none;">{tools.get('headline','')}</a>
    </h2>
    <p style="font-size:17px;color:#b0b0c8;line-height:1.8;margin:0 0 22px;">
      {tools.get('body','')}
    </p>
    <a href="{tools.get('url','#')}"
       style="display:inline-block;background:#ffb34718;color:#ffb347;
              border:1px solid #ffb34740;font-family:monospace;font-size:12px;
              letter-spacing:2px;font-weight:700;text-transform:uppercase;
              text-decoration:none;padding:12px 24px;border-radius:3px;">
      Check it out →
    </a>
  </div>
 
  <!-- ══ CLOSING THOUGHT ══ -->
  <div style="padding:30px 44px;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="width:4px;background:linear-gradient(180deg,#00d9b4,#7c6bff);
                   border-radius:2px;">&nbsp;</td>
        <td style="padding-left:20px;">
          <p style="font-family:Georgia,serif;font-size:17px;font-style:italic;
                    color:#8a8aaa;line-height:1.75;margin:0;">
            "{closing}"
          </p>
          <p style="font-size:13px;color:#4a4a6a;margin:10px 0 0;font-family:monospace;">
            — Dr. Prateek Singh
          </p>
        </td>
      </tr>
    </table>
  </div>
 
  {blogs_section_html}
 
  <!-- ══ CTA BANNER ══ -->
  <div style="padding:32px 44px;background:#0d0d1a;border-bottom:1px solid #1e1e35;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="vertical-align:middle;">
          <p style="font-family:Georgia,serif;font-size:18px;font-weight:700;
                    color:#f0f0f8;margin:0 0 4px;">Working on Healthcare AI?</p>
          <p style="font-size:14px;color:#6a6a8a;margin:0;">
            Let's discuss your project — free 30-min call.
          </p>
        </td>
        <td style="text-align:right;vertical-align:middle;">
          <a href="https://cal.com/prateek-singh-la8jpj"
             style="display:inline-block;background:#00d9b4;color:#08080f;
                    font-family:monospace;font-size:11px;letter-spacing:2px;
                    font-weight:700;text-transform:uppercase;text-decoration:none;
                    padding:12px 20px;border-radius:3px;white-space:nowrap;">
            Book a Call →
          </a>
        </td>
      </tr>
    </table>
  </div>
 
  <!-- ══ FOOTER ══ -->
  <div style="padding:32px 44px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;">
      <tr>
        <td>
          <p style="font-family:monospace;font-size:10px;letter-spacing:2px;
                    color:#00d9b4;text-transform:uppercase;margin:0 0 8px;">
            Dr. Prateek Singh
          </p>
          <p style="font-size:13px;color:#4a4a6a;line-height:1.8;margin:0;">
            Senior Manager, GenAI &amp; Digital Health AI<br>
            Samsung Research Institute, Noida · IIT Roorkee PhD
          </p>
        </td>
        <td style="text-align:right;vertical-align:middle;">
          <a href="https://prateeksinghphd.in"
             style="display:inline-block;background:#0d0d1a;color:#00d9b4;
                    border:1px solid #1e1e35;font-family:monospace;font-size:10px;
                    letter-spacing:2px;text-transform:uppercase;text-decoration:none;
                    padding:8px 14px;border-radius:3px;">
            My Blog →
          </a>
        </td>
      </tr>
    </table>
 
    <!-- Social links -->
    <div style="border-top:1px solid #1e1e35;padding-top:20px;
                display:flex;gap:16px;flex-wrap:wrap;">
      <a href="https://prateeksinghphd.in"
         style="font-size:13px;color:#4a4a6a;text-decoration:none;">🌐 Website</a>
      <a href="https://www.linkedin.com/in/prateek29s/"
         style="font-size:13px;color:#4a4a6a;text-decoration:none;">💼 LinkedIn</a>
      <a href="https://scholar.google.com/citations?user=nYZhJaMAAAAJ&hl=en"
         style="font-size:13px;color:#4a4a6a;text-decoration:none;">📚 Scholar</a>
      <a href="https://cal.com/prateek-singh-la8jpj"
         style="font-size:13px;color:#4a4a6a;text-decoration:none;">📅 Book a Call</a>
    </div>
 
    <p style="margin:24px 0 0;">
      <a href="{unsubscribe_url}"
         style="font-family:monospace;font-size:10px;letter-spacing:2px;
                color:#2a2a4a;text-decoration:none;text-transform:uppercase;">
        Unsubscribe · One click, no questions asked
      </a>
    </p>
  </div>
 
</div>
</body>
</html>"""
 
 
# ══════════════════════════════════════════════════════════════════
# STEP 5: SEND EMAILS VIA RESEND
# ══════════════════════════════════════════════════════════════════
 
def send_newsletter(emails: list[str], digest: dict, date_str: str, blogs: list = None) -> dict:
    """Send newsletter to all subscribers via Resend."""
    results = {'sent': 0, 'failed': 0, 'errors': []}
 
    log.info(f"Sending to {len(emails)} subscribers...")
 
    for i, email in enumerate(emails):
        try:
            html    = build_email_html(digest, date_str, email, blogs=blogs)
            subject = digest.get('subject', f'🧠 AI Daily — {date_str}')
 
            r = requests.post(
                'https://api.resend.com/emails',
                headers={
                    'Authorization': f'Bearer {RESEND_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'from': f'{FROM_NAME} <{FROM_EMAIL}>',
                    'to': email,
                    'subject': subject,
                    'html': html
                },
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
 
            # Rate limiting — Resend free tier: 2 req/sec
            time.sleep(0.6)
 
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({'email': email, 'error': str(e)})
            log.error(f"  Exception for {email}: {e}")
 
    return results
 
 
# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
 
def main():
    date_str = datetime.now(timezone.utc).strftime('%B %d, %Y')
    log.info(f"{'='*60}")
    log.info(f"Daily AI Newsletter Agent — {date_str}")
    if TEST_MODE:
        log.info("⚠️  TEST MODE — emails only sent to TEST_EMAIL")
    log.info(f"{'='*60}")
 
    # Step 1: Fetch news
    news_items = fetch_all_news()
    if not news_items:
        log.error("No news items fetched — aborting")
        return
 
    # Step 1.5: Fetch my latest blog posts
    my_blogs = fetch_my_blogs(max_posts=3)
 
    # Step 2: Write digest
    digest = write_digest_with_llm(news_items, date_str)
 
    # Step 3: Get subscribers
    emails = fetch_subscribers()
    if not emails:
        log.error("No subscribers found — aborting")
        return
 
    # Step 4: Send
    results = send_newsletter(emails, digest, date_str, blogs=my_blogs)
 
    # Summary
    log.info(f"{'='*60}")
    log.info(f"✅ Sent:   {results['sent']}")
    log.info(f"❌ Failed: {results['failed']}")
    if results['errors']:
        log.warning(f"Errors: {json.dumps(results['errors'], indent=2)}")
    log.info(f"{'='*60}")
 
 
if __name__ == '__main__':
    main()
 
