#!/usr/bin/env python3
import os
import sys
import json
import re
import time
import concurrent.futures
from urllib.parse import urlparse
from typing import Optional, List

import requests
import google.genai as genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize environment variables
load_dotenv()

# Global variables/clients
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
MODEL = "gemini-3.1-flash-lite"

MIN_ACCEPTABLE_SCORE = 0.6
MAX_QUERY_ATTEMPTS = 3

# Initialize FastAPI App
app = FastAPI(
    title="Query Selector & Hashtag Explorer API",
    description="API for generating search queries, ranking them, and extracting Instagram creators/hashtags."
)

# ---------------------------------------------------------------------------
# Pydantic Schemas for API Requests
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    prompt: str
    scoring_mode: Optional[str] = "heuristic"
    profile_pages: Optional[int] = 1
    hashtag_pages: Optional[int] = 5
    creator_pages: Optional[int] = 1

class HashtagRequest(BaseModel):
    hashtag: str
    pages: Optional[int] = 5

# ---------------------------------------------------------------------------
# Core Logic Helper Functions (Your Original Functions)
# ---------------------------------------------------------------------------
def generate_with_retry(max_retries: int = 4, base_delay: float = 2.0, **kwargs):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"  ! Gemini overloaded (attempt {attempt}/{max_retries}), retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  ! Gemini still unavailable after {max_retries} attempts.")
    raise last_error

def extract_entities(prompt: str) -> dict:
    system = (
        "You extract structured search entities from a user's request for finding "
        "social media creators/influencers. Return ONLY valid JSON, no preamble, "
        "no markdown fences. Schema:\n"
        "{\n"
        '  "platform": "instagram" | "youtube" | "linkedin" | "unspecified",\n'
        '  "niches": ["string", ...],\n'
        '  "locations": ["string", ...],\n'
        '  "tier": "micro" | "macro" | "mega" | "any",\n'
        '  "min_followers": number or null,\n'
        '  "gender": "male" | "female" | "both" | "unspecified"\n'
        "}"
    )
    resp = generate_with_retry(
        model=MODEL,
        contents=prompt,
        config={
            "system_instruction": system,
            "max_output_tokens": 500,
            "response_mime_type": "application/json",
        },
    )
    return json.loads(resp.text.strip())

def generate_queries(entities: dict, n: int = 5, avoid_queries: list = None) -> list:
    avoid_note = ""
    if avoid_queries:
        avoid_list = "\n".join(f"- {q}" for q in avoid_queries)
        avoid_note = (
            f"\n\nIMPORTANT: the following queries were already tried and did NOT "
            f"return good results (too few profile pages, too little relevance). "
            f"Do NOT repeat these or produce close variations of them — try "
            f"genuinely different angles, operators, or phrasing:\n{avoid_list}"
        )
    system = (
        f"You are an expert at Google advanced search operators (site:, intitle:, "
        f"inurl:, quotes, OR, minus-exclude). Given structured entities describing "
        f"a search for social media creators, generate exactly {n} DISTINCT search "
        f"query strings. Each should test a different angle: e.g. bio-location "
        f"pattern matching, hashtag matching, listicle/press-mention matching, "
        f"profile-page-only filtering (excluding /p/ /reel/ /watch/ post URLs), "
        f"and a broad OR-combination.\n\n"
        f"OUTPUT FORMAT: return ONLY the {n} query strings, one per line, in "
        f"plain text. No numbering, no bullets, no markdown, no JSON, no "
        f"preamble, no explanation — just the raw query text on each line, "
        f"exactly as it should be typed into Google. IMPORTANT: every opening "
        f"parenthesis '(' must have a matching closing parenthesis ')' — "
        f"double-check each query is syntactically balanced before outputting it."
        f"{avoid_note}"
    )
    resp = generate_with_retry(
        model=MODEL,
        contents=json.dumps(entities),
        config={"system_instruction": system, "max_output_tokens": 800},
    )
    text = resp.text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = [re.sub(r"^(\d+[\.\)]\s*|[-*]\s*)", "", line) for line in lines]
    return cleaned[:n]

def fix_unbalanced_parens(query: str) -> str:
    open_count = query.count("(")
    close_count = query.count(")")
    if open_count > close_count:
        query = query + (")" * (open_count - close_count))
    return query

def run_search(query: str, num_results: int = 10, start: int = 0, max_retries: int = 3) -> list:
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set — see setup instructions at top of file.")
    query = fix_unbalanced_parens(query)
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "engine": "google",
        "start": start,
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get("https://serpapi.com/search", params=params, timeout=40)
            r.raise_for_status()
            data = r.json()
            return data.get("organic_results", [])
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 * attempt
                print(f"  ! Search timeout/error (attempt {attempt}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error

def run_search_paginated(query: str, num_pages: int = 5, per_page: int = 10, max_retries: int = 3) -> list:
    all_results = []
    for page in range(num_pages):
        start = page * per_page
        page_results = run_search(query, num_results=per_page, start=start, max_retries=max_retries)
        if not page_results:
            break
        all_results.extend(page_results)
    return all_results

EXCLUDED_PATH_MARKERS = ("/p/", "/reel/", "/tv/", "/stories/", "/watch", "/shorts")
REQUIRED_DOMAIN = "instagram.com"
REDIRECT_DOMAIN_MARKERS = ("google.com/goto", "google.com/url", "googleadservices.com")

def is_profile_like(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    full_url_lower = url.lower()
    if any(marker in full_url_lower for marker in REDIRECT_DOMAIN_MARKERS):
        return False
    if REQUIRED_DOMAIN not in domain:
        return False
    path = parsed.path
    if any(marker in path for marker in EXCLUDED_PATH_MARKERS):
        return False
    segments = [s for s in path.split("/") if s]
    return len(segments) <= 1

def is_instagram_result(url: str) -> bool:
    if not url:
        return False
    full_url_lower = url.lower()
    if any(marker in full_url_lower for marker in REDIRECT_DOMAIN_MARKERS):
        return False
    domain = urlparse(url).netloc.lower()
    return REQUIRED_DOMAIN in domain

def keyword_density(text: str, keywords: list) -> float:
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)

def score_query_results(results: list, keywords: list) -> dict:
    if not results:
        return {"score": 0.0, "profile_ratio": 0.0, "avg_keyword_density": 0.0, "count": 0}
    profile_flags = [is_profile_like(r.get("link", "")) for r in results]
    profile_ratio = sum(profile_flags) / len(results)
    densities = [
        keyword_density(f"{r.get('title', '')} {r.get('snippet', '')}", keywords)
        for r in results
    ]
    avg_density = sum(densities) / len(densities)
    count_score = min(len(results) / 10, 1.0)
    composite = (0.5 * profile_ratio) + (0.3 * avg_density) + (0.2 * count_score)
    return {
        "score": round(composite, 3),
        "profile_ratio": round(profile_ratio, 2),
        "avg_keyword_density": round(avg_density, 2),
        "count": len(results),
    }

def score_query_results_llm(results: list, entities: dict, original_prompt: str) -> dict:
    if not results:
        return {"score": 0.0, "avg_rating": 0.0, "count": 0, "ratings": []}
    listing = "\n".join(
        f"{i+1}. Title: {r.get('title', '')}\n   Snippet: {r.get('snippet', '')}\n   URL: {r.get('link', '')}"
        for i, r in enumerate(results)
    )
    system = (
        "You are a strict, expert influencer/creator scout. Given a search brief "
        "and a numbered list of search results, rate EACH result 1-10 on how well "
        "it represents a real, matching creator profile (not a random article, "
        "spam page, or single post). Consider: does it look like an actual "
        "profile/channel page, does it plausibly match the location and niche, "
        "and does it look like a real, active account rather than noise.\n\n"
        "OUTPUT FORMAT: return ONLY the ratings, one per line, in the format "
        "'N: score' (e.g. '1: 7'), one line per result, no other text."
    )
    user_content = (
        f"BRIEF: {original_prompt}\n"
        f"ENTITIES: {json.dumps(entities)}\n\n"
        f"RESULTS:\n{listing}"
    )
    resp = generate_with_retry(
        model=MODEL,
        contents=user_content,
        config={"system_instruction": system, "max_output_tokens": 300},
    )
    text = resp.text.strip()
    ratings = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)", line)
        if match:
            ratings.append(float(match.group(2)))
    if not ratings:
        return {"score": 0.0, "avg_rating": 0.0, "count": len(results), "ratings": []}
    avg_rating = sum(ratings) / len(ratings)
    normalized_score = avg_rating / 10
    return {
        "score": round(normalized_score, 3),
        "avg_rating": round(avg_rating, 2),
        "count": len(results),
        "ratings": ratings,
    }

HASHTAG_RE = re.compile(r"#\w+")

def extract_hashtags_from_results(results: list, top_n: int = 10, exclude_tag: str = None, context_label: str = "") -> list:
    if not results:
        return []
    relevant_results = [r for r in results if is_instagram_result(r.get("link", ""))]
    if not relevant_results:
        return extract_hashtags_via_llm(context_label, results, top_n)

    exclude_lower = exclude_tag.lower() if exclude_tag else None
    hashtag_counts = {}
    for r in relevant_results:
        text = f"{r.get('title', '')} {r.get('snippet', '')}"
        found = HASHTAG_RE.findall(text)
        for ht in found:
            ht_lower = ht.lower()
            if exclude_lower and ht_lower == exclude_lower:
                continue
            hashtag_counts[ht] = hashtag_counts.get(ht, 0) + 1

    if not hashtag_counts:
        return extract_hashtags_via_llm(context_label, relevant_results, top_n)

    sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: (-x[1], x[0]))
    seen = set()
    unique_hashtags = []
    for ht, count in sorted_hashtags:
        key = ht.lower()
        if key not in seen:
            seen.add(key)
            unique_hashtags.append((ht, count))

    if len(unique_hashtags) < top_n:
        needed = top_n - len(unique_hashtags)
        llm_extra = extract_hashtags_via_llm(context_label, relevant_results, needed)
        existing = {ht.lower() for ht, _ in unique_hashtags}
        for ht, count in llm_extra:
            if ht.lower() not in existing:
                unique_hashtags.append((ht, count))
                existing.add(ht.lower())
    return unique_hashtags[:top_n]

def extract_hashtags_via_llm(context_label: str, results: list, top_n: int = 10) -> list:
    listing = "\n".join(f"{i+1}. {r.get('title', '')} — {r.get('snippet', '')}" for i, r in enumerate(results[:15]))
    system = (
        f"You are an Instagram hashtag expert. Given a list of Google search results "
        f"related to '{context_label}', extract or suggest the most relevant and "
        f"popular related Instagram hashtags. Focus on hashtags that would actually "
        f"be used on Instagram posts in the same niche.\n\n"
        f"OUTPUT FORMAT: return ONLY a JSON array of strings, like this:\n"
        f'["#hashtag1", "#hashtag2", "#hashtag3", ...]\n'
        f"Return exactly {top_n} hashtags. No other text, no markdown fences."
    )
    user_content = f"Context: {context_label}\n\nGoogle search results:\n{listing}"
    resp = generate_with_retry(model=MODEL, contents=user_content, config={"system_instruction": system, "max_output_tokens": 300})
    try:
        hashtags = json.loads(resp.text.strip())
        if isinstance(hashtags, list):
            return [(ht, 1) for ht in hashtags[:top_n]]
    except (json.JSONDecodeError, TypeError):
        pass
    return []

def find_related_hashtags(hashtag: str, top_n: int = 10, num_pages: int = 5) -> list:
    tag = hashtag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag
    query = f'site:instagram.com "{tag}"'
    results = run_search_paginated(query, num_pages=num_pages, per_page=10)
    if not results:
        return []
    return extract_hashtags_from_results(results, top_n=top_n, exclude_tag=tag, context_label=tag)

USERNAME_MENTION_RE = re.compile(r"@([A-Za-z0-9_.]{1,30})")

def extract_creator_from_result(result: dict) -> dict:
    link = result.get("link", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    if not is_instagram_result(link):
        return None
    if is_profile_like(link):
        segments = [s for s in urlparse(link).path.split("/") if s]
        if not segments:
            return None
        username = segments[0]
        return {
            "username": username,
            "profile_url": f"https://www.instagram.com/{username}/",
            "source_title": title,
            "found_via": "profile_link",
        }
    match = USERNAME_MENTION_RE.search(f"{title} {snippet}")
    if match:
        username = match.group(1)
        return {
            "username": username,
            "profile_url": f"https://www.instagram.com/{username}/",
            "source_title": title,
            "found_via": "mention",
        }
    return None

def fetch_creators_for_hashtag(hashtag: str, num_pages: int = 1, locations: list = None, niches: list = None) -> list:
    tag = hashtag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag
    scoping_keywords = (locations or []) + (niches or [])
    query = f'site:instagram.com "{tag}"'
    if scoping_keywords:
        query += " (" + " OR ".join(f'"{kw}"' for kw in scoping_keywords) + ")"
    results = run_search_paginated(query, num_pages=num_pages, per_page=10)
    require_keywords = locations if locations else niches
    seen_usernames = set()
    creators = []
    for r in results:
        creator = extract_creator_from_result(r)
        if not creator:
            continue
        if require_keywords:
            text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
            if not any(kw.lower() in text for kw in require_keywords):
                continue
        key = creator["username"].lower()
        if key in seen_usernames:
            continue
        seen_usernames.add(key)
        creator["hashtag"] = tag
        creators.append(creator)
    return creators

def fetch_creators_for_hashtags(hashtags: list, num_pages: int = 1, locations: list = None, niches: list = None) -> dict:
    tags = [h[0] if isinstance(h, tuple) else h for h in hashtags]
    results_by_tag = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tags) or 1)) as executor:
        future_map = {
            executor.submit(fetch_creators_for_hashtag, tag, num_pages, locations, niches): tag
            for tag in tags
        }
        for future in concurrent.futures.as_completed(future_map):
            tag = future_map[future]
            try:
                results_by_tag[tag] = future.result()
            except Exception as e:
                results_by_tag[tag] = []
    return {tag: results_by_tag.get(tag, []) for tag in tags}

def execute_search_pipeline(prompt: str, scoring_mode: str, profile_pages: int, hashtag_pages: int, creator_pages: int):
    entities = extract_entities(prompt)
    keywords = entities.get("niches", []) + entities.get("locations", [])
    all_tried_queries = []
    scored = []
    attempt = 1

    while attempt <= MAX_QUERY_ATTEMPTS:
        if attempt == 1:
            queries = generate_queries(entities, n=5)
        else:
            queries = generate_queries(entities, n=5, avoid_queries=all_tried_queries)

        all_tried_queries.extend(queries)
        results_by_query = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {executor.submit(run_search_paginated, q, profile_pages, 10): q for q in queries}
            for future in concurrent.futures.as_completed(future_map):
                q = future_map[future]
                try:
                    results_by_query[q] = future.result()
                except Exception:
                    results_by_query[q] = []

        this_round_scored = []
        for q, results in results_by_query.items():
            if scoring_mode == "llm":
                s = score_query_results_llm(results, entities, prompt)
            else:
                s = score_query_results(results, keywords)
            s["query"] = q
            s["results"] = results
            this_round_scored.append(s)

        scored.extend(this_round_scored)
        scored.sort(key=lambda x: x["score"], reverse=True)

        if scored[0]["score"] >= MIN_ACCEPTABLE_SCORE:
            break
        attempt += 1

    best = scored[0]
    hashtag_source_results = run_search_paginated(best["query"], num_pages=hashtag_pages, per_page=10)
    niche_label = ", ".join(entities.get("niches", []) or [prompt])
    niche_hashtags = extract_hashtags_from_results(hashtag_source_results, top_n=10, context_label=niche_label)
    
    creators_by_tag = fetch_creators_for_hashtags(
        niche_hashtags, num_pages=creator_pages, locations=entities.get("locations", []), niches=entities.get("niches", [])
    )
    
    all_creators = {}
    for tag, creators in creators_by_tag.items():
        for c in creators:
            all_creators.setdefault(c["username"].lower(), c)

    return {
        "entities": entities,
        "best_query": best["query"],
        "confidence": "OK" if best["score"] >= MIN_ACCEPTABLE_SCORE else "LOW-CONFIDENCE",
        "top_niche_hashtags": [ht for ht, _ in niche_hashtags],
        "creators_by_hashtag": creators_by_tag,
        "aggregated_unique_creators": list(all_creators.values())
    }

# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/search")
def run_influencer_search(payload: SearchRequest):
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("SERPAPI_KEY"):
        raise HTTPException(status_code=500, detail="Server environment keys missing (GEMINI_API_KEY / SERPAPI_KEY).")
    try:
        data = execute_search_pipeline(
            prompt=payload.prompt,
            scoring_mode=payload.scoring_mode,
            profile_pages=payload.profile_pages,
            hashtag_pages=payload.hashtag_pages,
            creator_pages=payload.creator_pages
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explore-hashtag")
def explore_hashtag(payload: HashtagRequest):
    if not os.environ.get("SERPAPI_KEY"):
        raise HTTPException(status_code=500, detail="SERPAPI_KEY is not configured on the server.")
    try:
        tag = payload.hashtag.strip()
        if not tag.startswith("#"):
            tag = "#" + tag
        related = find_related_hashtags(tag, top_n=10, num_pages=payload.pages)
        return {"status": "success", "hashtag": tag, "related_hashtags": [ht for ht, _ in related]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API operational. Visit /docs for the Swagger interactive control panel."}