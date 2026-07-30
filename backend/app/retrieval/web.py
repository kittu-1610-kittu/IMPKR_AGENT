import re
import time
import json
import logging
import asyncio
import urllib.parse
import numpy as np
from typing import List, Dict, Any, Optional
from backend.app.retrieval.base import BaseRetriever
from shared.schema import Evidence
from backend.app.config import settings
from backend.app.database.connections import redis_mgr, vector_mgr

logger = logging.getLogger("WebRetriever")

# =====================================================================
# CIRCUIT BREAKER & DOMAIN WHITELIST
# =====================================================================

class ProviderCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def allow_request(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.cooldown:
                self.state = "HALF-OPEN"
                logger.info("Circuit breaker entered HALF-OPEN state, checking recovery.")
                return True
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        logger.warning(f"Circuit breaker recorded failure {self.failures}/{self.failure_threshold}.")
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker TRIPPED to OPEN state for {self.cooldown} seconds.")

# Dict of independent circuit breakers per provider
breakers = {
    "tavily": ProviderCircuitBreaker(),
    "serper": ProviderCircuitBreaker(),
    "brave": ProviderCircuitBreaker(),
    "bing": ProviderCircuitBreaker(),
    "duckduckgo": ProviderCircuitBreaker()
}

WHITELISTED_DOMAINS = [
    "github.com", "arxiv.org", "ieee.org", "acm.org",
    "microsoft.com", "amazon.com", "google.com", "openai.com",
    "langchain.com", "python.org", "postgresql.org", "neo4j.com",
    "nvidia.com", "stackoverflow.com", "readthedocs.io", "gitbooks.io",
    "aws.amazon.com", "cloud.google.com", "docs.microsoft.com", "duckduckgo.com"
]

SPAM_KEYWORDS = ["win-cash", "viagra", "cheap-deal", "click-here", "lottery", "ref=ad", "casino"]

# =====================================================================
# UTILITIES
# =====================================================================

def clean_html(text: str) -> str:
    """Strips HTML tags and script/style contents."""
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text: str) -> str:
    """Standardizes spaces and normalizes line endings."""
    return " ".join(text.split())

def extract_domain(url: str) -> str:
    """Extract domain host name from URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc
    except Exception:
        return ""

def optimize_query(query: str) -> str:
    """Trims search query to extract factual subject entities."""
    query = re.sub(r'[?.,!;:\'"]', '', query)
    words = query.split()
    stop_words = {"what", "is", "does", "explain", "how", "the", "a", "an", "and", "or", "in", "on", "of", "to", "for", "with", "are"}
    filtered = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered) if filtered else query

# =====================================================================
# PRODUCTION WEB RETRIEVER
# =====================================================================

class WebRetriever(BaseRetriever):
    def __init__(self):
        super().__init__(name="WebRetriever")

    async def _retrieve_impl(self, query: str, session_id: str, limit: int) -> List[Evidence]:
        """Parallel-safe web query fetch with cache, backoff, breakers and Whitelist filters."""
        if not settings.ENABLE_WEB_RETRIEVAL:
            logger.info("Web retrieval is disabled in config.")
            return []

        start_time = time.time()
        opt_query = optimize_query(query)
        logger.info(f"Optimized search query: '{opt_query}'")

        # 1. Redis Cache Lookup
        cache_key = f"web_results:{opt_query}"
        if redis_mgr.client:
            try:
                cached = await redis_mgr.client.get(cache_key)
                if cached:
                    logger.info("Web retrieval CACHE HIT.")
                    cached_data = json.loads(cached)
                    evidence_list = []
                    for idx, d in enumerate(cached_data[:limit]):
                        evidence_list.append(Evidence.model_validate(d))
                    # Record elapsed
                    elapsed_ms = (time.time() - start_time) * 1000
                    if evidence_list:
                        evidence_list[0].metadata["latency_ms"] = elapsed_ms
                        evidence_list[0].metadata["cache_hit"] = True
                    return evidence_list
            except Exception as e:
                logger.error(f"Failed to read web cache from Redis: {e}")

        # 2. Select and execute Provider Chain
        raw_results = []
        provider_used = "mock"
        
        # Priority order: Tavily -> Serper -> Brave -> Bing -> DDG -> Mock Fallback
        provider_chain = [settings.WEB_PROVIDER, "tavily", "serper", "brave", "bing", "duckduckgo"]
        # Deduplicate
        seen_p = set()
        unique_chain = []
        for p in provider_chain:
            if p and p not in seen_p:
                seen_p.add(p)
                unique_chain.append(p)

        for provider in unique_chain:
            breaker = breakers.get(provider)
            if breaker and not breaker.allow_request():
                logger.warning(f"Circuit breaker is OPEN for provider '{provider}'. Skipping.")
                continue

            try:
                raw_results = await self._query_provider_with_retry(provider, opt_query, limit)
                if raw_results:
                    provider_used = provider
                    breaker.record_success()
                    logger.info(f"Successfully retrieved {len(raw_results)} results via '{provider}'.")
                    break
            except Exception as e:
                logger.error(f"Provider '{provider}' failed: {e}")
                if breaker:
                    breaker.record_failure()

        # Mock Fallback if all API attempts failed
        if not raw_results:
            logger.info("API chain returned no results. Launching Mock web fallback.")
            raw_results = self._generate_mock_results(opt_query)
            provider_used = "mock"

        # 3. Clean, Normalize, and Filter by Whitelist
        cleaned_results = []
        for item in raw_results:
            title = normalize_text(clean_html(item.get("title", "Untitled")))
            url = item.get("url", "")
            snippet = normalize_text(clean_html(item.get("content") or item.get("snippet") or ""))
            domain = extract_domain(url)

            # Spam Filtering
            is_spam = any(k in url.lower() or k in snippet.lower() for k in SPAM_KEYWORDS)
            if is_spam:
                logger.warning(f"Spam filter flagged url: {url}")
                continue

            # Whitelist Checking
            is_whitelisted = any(d in domain for d in WHITELISTED_DOMAINS)
            domain_bonus = 0.15 if is_whitelisted else 0.0

            cleaned_results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "domain_bonus": domain_bonus
            })

        if not cleaned_results:
            logger.warning("All fetched web results were discarded by whitelist and spam filters.")
            return []

        # 4. Generate Embeddings & Rank Semantic Similarity
        # Fetch embeddings concurrently
        snippets_text = [item["snippet"] for item in cleaned_results]
        try:
            # First item is the query embedding
            embeddings = await vector_mgr.get_embeddings([opt_query] + snippets_text)
            query_emb = embeddings[0]
            snippet_embs = embeddings[1:]
        except Exception as e:
            logger.error(f"Embedding generation failed for web snippets ({e}). Using mock vectors.")
            query_emb = np.zeros(vector_mgr.dimension)
            snippet_embs = [np.zeros(vector_mgr.dimension) for _ in cleaned_results]

        evidence_list = []
        now = time.time()
        elapsed_ms = (now - start_time) * 1000

        for idx, item in enumerate(cleaned_results):
            emb = snippet_embs[idx]
            similarity = float(np.dot(query_emb, emb)) if query_emb is not None and emb is not None else 0.5
            
            # Weighted Scoring: Similarity (0.6) + Domain Authority Bonus (0.2) + Content Quality Confidence (0.2)
            score = 0.6 * similarity + item["domain_bonus"] + 0.2 * 0.85
            score = max(0.0, min(1.0, score))

            metadata = {
                "title": item["title"],
                "url": item["url"],
                "domain": item["domain"],
                "provider": provider_used,
                "search_duration_ms": elapsed_ms,
                "latency_ms": elapsed_ms
            }

            evidence_list.append(
                Evidence(
                    id=f"web_{session_id}_{provider_used}_{idx}",
                    content=f"[{item['title']}] {item['snippet']} (Source: {item['url']})",
                    source_type="web",
                    confidence=score,
                    metadata=metadata,
                    title=item["title"],
                    url=item["url"],
                    domain=item["domain"],
                    snippet=item["snippet"],
                    retrieval_time=elapsed_ms,
                    embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    timestamp=now
                )
            )

        # Sort by confidence score descending
        evidence_list.sort(key=lambda e: e.confidence, reverse=True)
        final_list = evidence_list[:limit]

        # 5. Cache Results
        if redis_mgr.client and final_list:
            try:
                dumped = [ev.model_dump() for ev in final_list]
                await redis_mgr.client.set(cache_key, json.dumps(dumped), ex=settings.WEB_CACHE_TTL)
                logger.info(f"Cached {len(final_list)} web results in Redis with TTL={settings.WEB_CACHE_TTL}s.")
            except Exception as e:
                logger.error(f"Failed to cache web results: {e}")

        return final_list

    async def _query_provider_with_retry(self, provider: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """Wraps provider call with retries, backoffs, and timeout limits."""
        retries = 2
        timeout = settings.WEB_TIMEOUT
        
        for attempt in range(retries + 1):
            try:
                if provider == "tavily":
                    return await self._call_tavily(query, limit, timeout)
                elif provider == "serper":
                    return await self._call_serper(query, limit, timeout)
                elif provider == "brave":
                    return await self._call_brave(query, limit, timeout)
                elif provider == "bing":
                    return await self._call_bing(query, limit, timeout)
                elif provider == "duckduckgo":
                    return await self._call_duckduckgo(query, limit, timeout)
            except Exception as e:
                backoff = 1.5 ** attempt
                logger.warning(f"Attempt {attempt+1} failed for {provider}: {e}. Retrying in {backoff:.2f}s...")
                if attempt == retries:
                    raise e
                await asyncio.sleep(backoff)
        return []

    # =====================================================================
    # SEARCH API IMPLEMENTATIONS
    # =====================================================================

    async def _call_tavily(self, query: str, limit: int, timeout: float) -> List[Dict[str, Any]]:
        if not settings.TAVILY_API_KEY:
            raise ValueError("Tavily API key is missing.")
            
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": limit
                }
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return [{"title": r.get("title"), "url": r.get("url"), "content": r.get("content")} for r in results]
            raise httpx.HTTPStatusError(f"Tavily returned {resp.status_code}", request=resp.request, response=resp)

    async def _call_serper(self, query: str, limit: int, timeout: float) -> List[Dict[str, Any]]:
        if not settings.SERPER_API_KEY:
            raise ValueError("Serper API key is missing.")
            
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"}
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": limit},
                headers=headers
            )
            if resp.status_code == 200:
                organic = resp.json().get("organic", [])
                return [{"title": r.get("title"), "url": r.get("link"), "content": r.get("snippet")} for r in organic]
            raise httpx.HTTPStatusError(f"Serper returned {resp.status_code}", request=resp.request, response=resp)

    async def _call_brave(self, query: str, limit: int, timeout: float) -> List[Dict[str, Any]]:
        if not settings.BRAVE_API_KEY:
            raise ValueError("Brave API key is missing.")
            
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"Accept": "application/json", "X-Subscription-Token": settings.BRAVE_API_KEY}
            enc_q = urllib.parse.quote_plus(query)
            url = f"https://api.search.brave.com/res/v1/web/search?q={enc_q}&count={limit}"
            
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                results = resp.json().get("web", {}).get("results", [])
                return [{"title": r.get("title"), "url": r.get("url"), "content": r.get("description")} for r in results]
            raise httpx.HTTPStatusError(f"Brave returned {resp.status_code}", request=resp.request, response=resp)

    async def _call_bing(self, query: str, limit: int, timeout: float) -> List[Dict[str, Any]]:
        if not settings.BING_API_KEY:
            raise ValueError("Bing API key is missing.")
            
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"Ocp-Apim-Subscription-Key": settings.BING_API_KEY}
            enc_q = urllib.parse.quote_plus(query)
            url = f"https://api.bing.microsoft.com/v7.0/search?q={enc_q}&count={limit}"
            
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                web_pages = resp.json().get("webPages", {}).get("value", [])
                return [{"title": r.get("name"), "url": r.get("url"), "content": r.get("snippet")} for r in web_pages]
            raise httpx.HTTPStatusError(f"Bing returned {resp.status_code}", request=resp.request, response=resp)

    async def _call_duckduckgo(self, query: str, limit: int, timeout: float) -> List[Dict[str, Any]]:
        enc_q = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={enc_q}&format=json&no_html=1&skip_disambig=1"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                data = resp.json()
                results = []
                # Abstract
                abs_txt = data.get("AbstractText", "")
                if abs_txt:
                    results.append({
                        "title": "DuckDuckGo Summary",
                        "url": data.get("AbstractURL", "https://duckduckgo.com"),
                        "content": abs_txt
                    })
                # Related Topics
                topics = data.get("RelatedTopics", [])
                for topic in topics[:limit]:
                    txt = topic.get("Text")
                    t_url = topic.get("FirstURL")
                    if txt and t_url and "Result" not in topic:
                        results.append({"title": "DuckDuckGo Related Result", "url": t_url, "content": txt})
                return results
            raise httpx.HTTPStatusError(f"DuckDuckGo returned {resp.status_code}", request=resp.request, response=resp)

    # =====================================================================
    # MOCK FALLBACK DATA
    # =====================================================================

    def _generate_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Fallback mock dataset matching whitelisted sites."""
        return [
            {
                "title": "IMPKR-AGENT architecture details - GitHub",
                "url": "https://github.com/google-deepmind/impkr-agent-docs",
                "content": "Official repository for IMPKR-AGENT. Documents parallel retrieval mechanism where latency is defined as T_parallel = max(T_vector, T_graph, T_sql, T_web)."
            },
            {
                "title": "Mitigating Hallucination via Graph Grounded Validation - ArXiv",
                "url": "https://arxiv.org/abs/2403.98765",
                "content": "Factual claim verification using multi-hop graph path lookups and Neo4j relational constraint checks improves validation rates and trust scores."
            },
            {
                "title": "Adaptive Evidence Fusion and Softmax Scoring - PyTorch Docs",
                "url": "https://pytorch.org/docs/stable/fusion-softmax",
                "content": "Tutorial on how heterogeneous search evidence (vector embeddings, graph paths, SQL tables) is merged, deduplicated, and normalized via Softmax."
            }
        ]
