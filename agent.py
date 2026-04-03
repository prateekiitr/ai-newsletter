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
FROM_EMAIL      = os.environ.get('FROM_EMAIL', 'newsletter@prateeksinghphd.in')
FROM_NAME       = os.environ.get('FROM_NAME', 'Dr. Prateek Singh')
MODEL           = 'llama-3.1-70b-versatile'  # Hermes-3 on Groq — change if available
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

    log.info("Calling Groq API (Hermes-3)...")

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 1200,
        'response_format': {'type': 'json_object'}
    }

    r = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers,
        json=payload,
        timeout=30
    )
    r.raise_for_status()

    content = r.json()['choices'][0]['message']['content']
    digest  = json.loads(content)
    log.info(f"Digest written. Subject: {digest.get('subject', 'N/A')}")
    return digest


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
        r = requests.get(
            SUBSCRIBERS_URL,
            headers={'Authorization': f'Bearer {ADMIN_TOKEN}'},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        emails = [s['email'] for s in data.get('subscribers', []) if s.get('active', True)]
        log.info(f"Found {len(emails)} active subscribers")
        return emails
    except Exception as e:
        log.error(f"Failed to fetch subscribers: {e}")
        raise


# ══════════════════════════════════════════════════════════════════
# STEP 4: BUILD HTML EMAIL
# ══════════════════════════════════════════════════════════════════

def build_email_html(digest: dict, date_str: str, email: str) -> str:
    """Build the full HTML email from the digest."""

    unsubscribe_url = f"https://prateeksinghphd.in/api/unsubscribe?email={requests.utils.quote(email)}"

    papers_html = ''.join([
        f"""<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:12px;">
              <span style="color:#00d9b4;flex-shrink:0;font-size:12px;margin-top:2px;">▸</span>
              <div>
                <a href="{p['url']}" style="color:#e0e0f0;text-decoration:none;font-weight:500;font-size:14px;">{p['title']}</a>
                <p style="color:#8a8aaa;font-size:13px;margin:3px 0 0;">{p['summary']}</p>
              </div>
            </div>"""
        for p in digest.get('papers', [])
    ])

    top     = digest.get('top_story', {})
    health  = digest.get('healthcare_spotlight', {})
    tools   = digest.get('tools_repos', {})
    closing = digest.get('closing_thought', '')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{digest.get('subject', 'Daily AI Newsletter')}</title>
</head>
<body style="margin:0;padding:0;background:#0a0a0f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">

  <div style="max-width:600px;margin:0 auto;padding:20px 0 40px;">

    <!-- Header -->
    <div style="padding:32px 40px 24px;border-bottom:1px solid #2a2a3f;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:4px;color:#00d9b4;
                text-transform:uppercase;margin:0 0 8px;">Daily AI Newsletter</p>
      <p style="color:#6a6a8a;font-size:12px;margin:0;font-family:monospace;">
        {date_str} &nbsp;·&nbsp; 9:00 AM IST
      </p>
    </div>

    <!-- Top Story -->
    <div style="padding:28px 40px;border-bottom:1px solid #1a1a2f;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#00d9b4;
                text-transform:uppercase;margin:0 0 14px;">
        ── Top Story
      </p>
      <h2 style="font-size:20px;font-weight:800;color:#f0f0f8;margin:0 0 10px;line-height:1.3;">
        <a href="{top.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{top.get('headline','')}</a>
      </h2>
      <p style="color:#b0b0c8;font-size:14px;line-height:1.75;margin:0 0 14px;">
        {top.get('body','')}
      </p>
      <a href="{top.get('url','#')}" style="font-family:monospace;font-size:11px;
               letter-spacing:2px;color:#00d9b4;text-transform:uppercase;text-decoration:none;">
        Read more →
      </a>
    </div>

    <!-- Healthcare AI Spotlight -->
    <div style="padding:28px 40px;border-bottom:1px solid #1a1a2f;background:#0d0d18;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#ff6b9d;
                text-transform:uppercase;margin:0 0 14px;">
        ── Healthcare AI Spotlight
      </p>
      <h2 style="font-size:18px;font-weight:700;color:#f0f0f8;margin:0 0 10px;line-height:1.3;">
        <a href="{health.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{health.get('headline','')}</a>
      </h2>
      <p style="color:#b0b0c8;font-size:14px;line-height:1.75;margin:0 0 14px;">
        {health.get('body','')}
      </p>
      <a href="{health.get('url','#')}" style="font-family:monospace;font-size:11px;
               letter-spacing:2px;color:#ff6b9d;text-transform:uppercase;text-decoration:none;">
        Read more →
      </a>
    </div>

    <!-- Papers -->
    <div style="padding:28px 40px;border-bottom:1px solid #1a1a2f;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#7c6bff;
                text-transform:uppercase;margin:0 0 18px;">
        ── 3 Papers Worth Reading
      </p>
      {papers_html}
    </div>

    <!-- Tools & Repos -->
    <div style="padding:28px 40px;border-bottom:1px solid #1a1a2f;background:#0d0d18;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:3px;color:#ffb347;
                text-transform:uppercase;margin:0 0 14px;">
        ── Tool / Repo of the Day
      </p>
      <h3 style="font-size:16px;font-weight:700;color:#f0f0f8;margin:0 0 8px;">
        <a href="{tools.get('url','#')}" style="color:#f0f0f8;text-decoration:none;">{tools.get('headline','')}</a>
      </h3>
      <p style="color:#b0b0c8;font-size:14px;line-height:1.75;margin:0 0 12px;">
        {tools.get('body','')}
      </p>
      <a href="{tools.get('url','#')}" style="font-family:monospace;font-size:11px;
               letter-spacing:2px;color:#ffb347;text-transform:uppercase;text-decoration:none;">
        Check it out →
      </a>
    </div>

    <!-- Closing thought -->
    <div style="padding:24px 40px;border-bottom:1px solid #1a1a2f;">
      <p style="color:#6a6a8a;font-size:13px;font-style:italic;line-height:1.7;margin:0;">
        💭 {closing}
      </p>
    </div>

    <!-- Footer -->
    <div style="padding:28px 40px;">
      <p style="color:#4a4a6a;font-size:12px;line-height:2;margin:0;">
        <strong style="color:#8a8aaa;">Dr. Prateek Singh</strong><br>
        Senior Manager, GenAI &amp; Digital Health AI · Samsung Research Institute, Noida<br>
        IIT Roorkee PhD · Healthcare AI · On-Device LLMs<br>
        <a href="https://prateeksinghphd.in" style="color:#00d9b4;text-decoration:none;">prateeksinghphd.in</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/prateek29s/" style="color:#4a4a6a;text-decoration:none;">LinkedIn</a>
        &nbsp;·&nbsp;
        <a href="https://cal.com/prateek-singh-la8jpj" style="color:#4a4a6a;text-decoration:none;">Book a call</a>
      </p>
      <p style="margin:16px 0 0;">
        <a href="{unsubscribe_url}"
           style="font-family:monospace;font-size:10px;letter-spacing:2px;
                  color:#3a3a5a;text-decoration:none;text-transform:uppercase;">
          Unsubscribe
        </a>
      </p>
    </div>

  </div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
# STEP 5: SEND EMAILS VIA RESEND
# ══════════════════════════════════════════════════════════════════

def send_newsletter(emails: list[str], digest: dict, date_str: str) -> dict:
    """Send newsletter to all subscribers via Resend."""
    results = {'sent': 0, 'failed': 0, 'errors': []}

    log.info(f"Sending to {len(emails)} subscribers...")

    for i, email in enumerate(emails):
        try:
            html    = build_email_html(digest, date_str, email)
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

    # Step 2: Write digest
    digest = write_digest_with_llm(news_items, date_str)

    # Step 3: Get subscribers
    emails = fetch_subscribers()
    if not emails:
        log.error("No subscribers found — aborting")
        return

    # Step 4: Send
    results = send_newsletter(emails, digest, date_str)

    # Summary
    log.info(f"{'='*60}")
    log.info(f"✅ Sent:   {results['sent']}")
    log.info(f"❌ Failed: {results['failed']}")
    if results['errors']:
        log.warning(f"Errors: {json.dumps(results['errors'], indent=2)}")
    log.info(f"{'='*60}")


if __name__ == '__main__':
    main()
