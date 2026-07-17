#!/usr/bin/env python3
"""
Query Generation & Selection System / Hashtag Explorer
=====================================================

Two modes:
  1. DEFAULT influencer search mode — generates 5 candidate queries,
     runs them through Google (SerpAPI), scores them, picks the best,
     AND extracts the top 10 related hashtags for that niche from the
     winning query's own results.
  2. --hashtags mode — given a hashtag like #fashion, searches Google
     for Instagram posts with that tag and extracts the top 10 related
     hashtags from the results.

SETUP
-----
1. Install dependencies:
     pip install google-genai requests --break-system-packages

2. Set environment variables:
     export GEMINI_API_KEY="your-gemini-key"
     export SERPAPI_KEY="your-serpapi-key"

USAGE
-----
  Influencer search (now also prints top 10 niche hashtags):
    python query_selector.py "lifestyle beauty influencers Bangalore"

  Hashtag exploration:
    python query_selector.py --hashtags "#fashion"

  LLM-as-judge scoring (influencer mode):
    python query_selector.py "lifestyle influencers" --llm
"""

import os
import sys
import json
import re
import time
import concurrent.futures
from urllib.parse import urlparse

import requests
import google.genai as genai
from google.genai import errors as genai_errors

from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
MODEL = "gemini-3.1-flash-lite"

MIN_ACCEPTABLE_SCORE = 0.6
MAX_QUERY_ATTEMPTS = 3


def generate_with_retry(max_retries: int = 4, base_delay: float = 2.0, **kwargs):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"  ! Gemini overloaded (attempt {attempt}/{max_retries}), "
                      f"retrying in {wait:.0f}s...")
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
                print(f"  ! Search timeout/error (attempt {attempt}/{max_retries}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def run_search_paginated(
    query: str,
    num_pages: int = 5,
    per_page: int = 10,
    max_retries: int = 3,
) -> list:
    all_results = []
    for page in range(num_pages):
        start = page * per_page
        page_results = run_search(
            query, num_results=per_page, start=start, max_retries=max_retries
        )
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
    """
    Lighter-weight check used for hashtag extraction (as opposed to
    is_profile_like, which is used for profile-search scoring).

    Hashtags actually live in post/reel captions, so unlike is_profile_like
    we deliberately do NOT reject /p/ /reel/ paths here — we only need to
    make sure the result is a real Instagram URL, not a non-Instagram
    domain or a Google redirect/tracking wrapper link that merely happened
    to mention a matching keyword.
    """
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


def extract_hashtags_from_results(
    results: list,
    top_n: int = 10,
    exclude_tag: str = None,
    context_label: str = "",
) -> list:
    if not results:
        return []

    # Only count hashtags from results that are actually real Instagram
    # URLs — this is what stops an unrelated result (wrong domain, or a
    # Google redirect wrapper) that merely contains a matching keyword from
    # polluting the hashtag counts with its own unrelated hashtags.
    relevant_results = [r for r in results if is_instagram_result(r.get("link", ""))]

    if not relevant_results:
        print("  No relevant Instagram results to extract hashtags from. "
              "Asking Gemini to infer related hashtags...")
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
        print("  No hashtags found directly in results. "
              "Asking Gemini to infer related hashtags...")
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
    listing = "\n".join(
        f"{i+1}. {r.get('title', '')} — {r.get('snippet', '')}"
        for i, r in enumerate(results[:15])
    )

    system = (
        f"You are an Instagram hashtag expert. Given a list of Google search results "
        f"related to '{context_label}', extract or suggest the most relevant and "
        f"popular related Instagram hashtags. Focus on hashtags that would actually "
        f"be used on Instagram posts in the same niche.\n\n"
        f"OUTPUT FORMAT: return ONLY a JSON array of strings, like this:\n"
        f'["#hashtag1", "#hashtag2", "#hashtag3", ...]\n'
        f"Return exactly {top_n} hashtags. No other text, no markdown fences."
    )

    user_content = (
        f"Context: {context_label}\n\n"
        f"Google search results:\n{listing}"
    )

    resp = generate_with_retry(
        model=MODEL,
        contents=user_content,
        config={"system_instruction": system, "max_output_tokens": 300},
    )

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

    print(f"\n{'='*70}")
    print(f"HASHTAG EXPLORER: {tag}")
    print(f"{'='*70}\n")

    query = f'site:instagram.com "{tag}"'
    print(f"Searching Google for: {query}")
    print(f"Fetching up to {num_pages} pages ({num_pages * 10} results)...\n")

    results = run_search_paginated(query, num_pages=num_pages, per_page=10)

    if not results:
        print("  ! No search results found.")
        return []

    print(f"Found {len(results)} search results across pages. "
          f"Extracting related hashtags...\n")

    return extract_hashtags_from_results(
        results, top_n=top_n, exclude_tag=tag, context_label=tag
    )


def run_hashtag_mode(hashtag: str, num_pages: int = 5):
    tag = hashtag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag

    related = find_related_hashtags(tag, top_n=10, num_pages=num_pages)
    print_hashtag_list(related, f"TOP 10 RELATED HASHTAGS FOR {tag}")


def print_hashtag_list(hashtags: list, header: str):
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}\n")

    if not hashtags:
        print("  No related hashtags found.")
    else:
        for i, (ht, count) in enumerate(hashtags, 1):
            print(f"  #{i:2d}  {ht}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CREATOR SCRAPING (find who's actually using each hashtag)
# ---------------------------------------------------------------------------

USERNAME_MENTION_RE = re.compile(r"@([A-Za-z0-9_.]{1,30})")


def extract_creator_from_result(result: dict) -> dict:
    """
    Given one search result, try to identify the Instagram creator behind it.

    Two cases:
      1. The result IS a real profile page (is_profile_like) -> take the
         username straight from the URL path.
      2. The result is a post/reel matching the hashtag (the common case for
         hashtag searches) -> look for an '@username' mention in the title
         or snippet, which Google's Instagram titles usually include
         (e.g. 'Jane Doe (@janedoe) on Instagram: ...').

    Returns None if no Instagram URL or no identifiable username is found.
    """
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


def fetch_creators_for_hashtag(
    hashtag: str,
    num_pages: int = 1,
    locations: list = None,
    niches: list = None,
) -> list:
    """
    Search Google for Instagram posts/profiles using the given hashtag and
    extract the unique creators found. Returns a list of creator dicts
    (deduplicated by username, case-insensitive).

    locations / niches (from the original search entities, e.g.
    locations=["Bangalore"], niches=["lifestyle", "beauty"]) scope this
    search. IMPORTANT: niche keywords alone are NOT a reliable filter here —
    a generic hashtag like #beauty often equals one of the niche keywords
    itself, so any big brand whose name contains that same word (e.g.
    "Fenty Beauty", "Chanel Beauty") trivially passes a niche-only check
    without being remotely local or relevant. Location is the actual
    discriminator between "a global brand that used #beauty" and "a
    Bangalore creator who used #beauty", so when locations are available we
    require a location match; niches are used only to scope the query, and
    as a weaker fallback filter when no location was given at all.
    """
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


def fetch_creators_for_hashtags(
    hashtags: list,
    num_pages: int = 1,
    locations: list = None,
    niches: list = None,
) -> dict:
    """
    Given a list of hashtags (strings, or (tag, count) tuples as produced by
    extract_hashtags_from_results), fetch creators for ALL of them in
    parallel. Returns {tag: [creator, ...]}.
    """
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
                print(f"  ! Creator fetch failed for {tag}: {e}")
                results_by_tag[tag] = []

    # preserve original hashtag order rather than as_completed order
    return {tag: results_by_tag.get(tag, []) for tag in tags}


def print_creators_by_hashtag(creators_by_tag: dict):
    print(f"\n{'='*70}")
    print("CREATORS FOUND PER HASHTAG")
    print(f"{'='*70}")

    all_creators = {}  # username(lower) -> creator dict, for the aggregated list
    for tag, creators in creators_by_tag.items():
        print(f"\n{tag}  ({len(creators)} creator(s) found)")
        if not creators:
            print("    (none found)")
        for c in creators:
            print(f"    @{c['username']}  —  {c['profile_url']}")
            all_creators.setdefault(c["username"].lower(), c)

    print(f"\n{'='*70}")
    print(f"AGGREGATED UNIQUE CREATORS ACROSS ALL HASHTAGS: {len(all_creators)}")
    print(f"{'='*70}")
    for c in sorted(all_creators.values(), key=lambda c: c["username"].lower()):
        print(f"  @{c['username']}  —  {c['profile_url']}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")

    return list(all_creators.values())


def main(
    prompt: str,
    scoring_mode: str = "heuristic",
    profile_pages: int = 1,
    hashtag_pages: int = 5,
    creator_pages: int = 1,
):
    print(f"\n{'='*70}\nPROMPT: {prompt}\n{'='*70}\n")
    print(f"Scoring mode: {scoring_mode}")
    print(f"Profile search depth: {profile_pages} page(s) per query "
          f"(up to {profile_pages * 10} results/query) — kept shallow for clean matches")
    print(f"Hashtag search depth: {hashtag_pages} page(s) on winning query "
          f"(up to {hashtag_pages * 10} results) — deeper for a well-sampled hashtag list\n")

    print("[1/6] Extracting entities...")
    entities = extract_entities(prompt)
    print(json.dumps(entities, indent=2))

    keywords = entities.get("niches", []) + entities.get("locations", [])

    all_tried_queries = []
    scored = []
    attempt = 1

    while attempt <= MAX_QUERY_ATTEMPTS:
        if attempt == 1:
            print(f"\n[2/6] Generating 5 candidate queries (attempt {attempt}/{MAX_QUERY_ATTEMPTS})...")
            queries = generate_queries(entities, n=5)
        else:
            print(f"\n[2/6] Best score so far ({scored[0]['score']}) is below the "
                  f"quality threshold ({MIN_ACCEPTABLE_SCORE}). Regenerating "
                  f"5 NEW candidate queries (attempt {attempt}/{MAX_QUERY_ATTEMPTS})...")
            queries = generate_queries(entities, n=5, avoid_queries=all_tried_queries)

        for i, q in enumerate(queries, 1):
            print(f"  Q{i}: {q}")
        all_tried_queries.extend(queries)

        print(f"\n[3/6] Running all 5 queries in parallel via search API "
              f"({profile_pages} page(s) each, kept shallow for profile quality)...")
        results_by_query = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(run_search_paginated, q, profile_pages, 10): q
                for q in queries
            }
            for future in concurrent.futures.as_completed(future_map):
                q = future_map[future]
                try:
                    results_by_query[q] = future.result()
                except Exception as e:
                    print(f"  ! Query failed: {q[:50]}... ({e})")
                    results_by_query[q] = []

        print(f"\n[4/6] Scoring each query's results ({scoring_mode})...")
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
    else:
        print(f"\n  ! Still below quality threshold ({MIN_ACCEPTABLE_SCORE}) after "
              f"{MAX_QUERY_ATTEMPTS} attempts. Proceeding with the best available "
              f"result — treat it as LOW-CONFIDENCE and sanity-check manually.")

    print(f"\n{'='*70}\nRANKED QUERIES (best {min(5, len(scored))} of "
          f"{len(scored)} tried across all attempts)\n{'='*70}")
    for i, s in enumerate(scored[:5], 1):
        if scoring_mode == "llm":
            print(f"\n#{i} — score {s['score']} "
                  f"(avg_rating={s.get('avg_rating', 0)}/10, "
                  f"results={s['count']})")
        else:
            print(f"\n#{i} — score {s['score']} "
                  f"(profile_ratio={s['profile_ratio']}, "
                  f"keyword_density={s['avg_keyword_density']}, "
                  f"results={s['count']})")
        print(f"   {s['query']}")


    best = scored[0]
    confidence_tag = (
        "LOW-CONFIDENCE — below quality threshold, sanity-check manually"
        if best["score"] < MIN_ACCEPTABLE_SCORE
        else "OK — meets quality threshold"
    )
    print(f"\n{'='*70}\nBEST QUERY SELECTED  [{confidence_tag}]\n{'='*70}")
    print(best["query"])
    print(f"\nTop {min(10, len(best['results']))} results from this query "
          f"(shallow search — clean profile matches):")
    for r in best["results"][:10]:
        rank = r.get("position", r.get("rank", "?"))
        print(f"  #{rank} {r.get('title', 'No title')}\n    {r.get('link', '')}")

    print(f"\n[5/6] Re-running the winning query with {hashtag_pages} page(s) "
          f"(up to {hashtag_pages * 10} results) specifically to build a "
          f"well-sampled hashtag list...")
    hashtag_source_results = run_search_paginated(
        best["query"], num_pages=hashtag_pages, per_page=10
    )
    print(f"  Fetched {len(hashtag_source_results)} results for hashtag sampling.")

    niche_label = ", ".join(entities.get("niches", []) or [prompt])
    niche_hashtags = extract_hashtags_from_results(
        hashtag_source_results, top_n=10, context_label=niche_label
    )
    print_hashtag_list(niche_hashtags, f"TOP 10 HASHTAGS FOR NICHE: {niche_label}")

    best["hashtags"] = niche_hashtags

    # ---- Creators: for EVERY extracted hashtag, find who's using it ----
    print(f"[6/6] Scraping creators who've used each of the "
          f"{len(niche_hashtags)} extracted hashtags "
          f"({creator_pages} page(s) per hashtag)...")
    creators_by_tag = fetch_creators_for_hashtags(
        niche_hashtags,
        num_pages=creator_pages,
        locations=entities.get("locations", []),
        niches=entities.get("niches", []),
    )
    all_creators = print_creators_by_hashtag(creators_by_tag)

    best["creators_by_hashtag"] = creators_by_tag
    best["creators"] = all_creators
    return best


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python query_selector.py "influencer search prompt" [--llm] '
              '[--profile-pages N] [--hashtag-pages N] [--creator-pages N]')
        print('  python query_selector.py --hashtags "#fashion" [--pages N]')
        print('  python query_selector.py --hashtags fashion     (# optional)')
        print()
        print("  --profile-pages N   pages fetched PER candidate query when picking")
        print("                      the best influencer-search query (10 results/page).")
        print("                      Default: 1. Kept shallow so results stay clean —")
        print("                      Google's page 1 has the least noise.")
        print("  --hashtag-pages N   pages fetched on ONLY the winning query, in a")
        print("                      separate deeper search used just to build the")
        print("                      hashtag list. Default: 5. More pages = better")
        print("                      sampled hashtags, since it draws from more posts.")
        print("  --creator-pages N   pages fetched PER extracted hashtag (10 in total)")
        print("                      when scraping creators who've used each one.")
        print("                      Default: 1. Raise this for a deeper creator list —")
        print("                      but note it multiplies searches by hashtag count.")
        print("  --pages N           (--hashtags mode only) pages to fetch. Default: 5.")
        sys.exit(1)

    args = sys.argv[1:]

    def pop_int_flag(flag_name, default):
        if flag_name in args:
            idx = args.index(flag_name)
            if idx + 1 < len(args):
                try:
                    value = int(args[idx + 1])
                except ValueError:
                    print(f"Error: {flag_name} must be an integer, e.g. {flag_name} 5")
                    sys.exit(1)
                del args[idx:idx + 2]
                return value
            else:
                print(f"Error: {flag_name} requires a number")
                sys.exit(1)
        return default

    hashtags_mode_pages = pop_int_flag("--pages", 5)
    profile_pages = pop_int_flag("--profile-pages", 1)
    hashtag_pages = pop_int_flag("--hashtag-pages", 5)
    creator_pages = pop_int_flag("--creator-pages", 1)

    if "--hashtags" in args:
        idx = args.index("--hashtags")
        if idx + 1 < len(args):
            hashtag = args[idx + 1]
        else:
            print("Error: --hashtags requires a hashtag argument, e.g. --hashtags '#fashion'")
            sys.exit(1)
        run_hashtag_mode(hashtag, num_pages=hashtags_mode_pages)
        sys.exit(0)

    mode = "heuristic"
    if "--llm" in args:
        mode = "llm"
        args.remove("--llm")

    user_prompt = " ".join(args)
    main(
        user_prompt,
        scoring_mode=mode,
        profile_pages=profile_pages,
        hashtag_pages=hashtag_pages,
        creator_pages=creator_pages,
    )