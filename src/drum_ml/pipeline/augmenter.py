import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from drum_ml.models.scaffolds import AugmentedRecord, PersonaType, ScaffoldRecord
from drum_ml.prompts.templates_personas import PERSONA_SYSTEM_PROMPTS


class MetrologyAugmenter:
    """Agent 3: Generates diverse, persona-conditioned user prompts using Multi-LLM APIs
    (Google Gemini, Anthropic Claude, OpenAI, local Ollama / LM Studio) with SQLite caching.
    """

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini/gemini-2.5-flash",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_db_path: str = "./data/cache/llm_cache.sqlite",
        temperature: float = 0.7,
        concurrency_limit: int = 5,
    ):
        self.provider = provider.lower()
        self.concurrency_limit = concurrency_limit
        
        # Normalize model identifier for LiteLLM providers
        if self.provider in ("google", "gemini") and not model.startswith("gemini/"):
            self.model = f"gemini/{model}"
        elif self.provider == "ollama" and not model.startswith("ollama/"):
            self.model = f"ollama/{model}"
        elif self.provider in ("openai", "vllm", "hosted_vllm") and not model.startswith("openai/"):
            self.model = f"openai/{model}"
        else:
            self.model = model

        # Default api_base for local providers if not specified
        if api_base:
            self.api_base = api_base
        elif self.provider == "ollama":
            self.api_base = "http://localhost:11434"
        elif self.provider == "lm_studio":
            self.api_base = "http://localhost:1234/v1"
        else:
            self.api_base = None

        if self.provider == "ollama" and self.api_base:
            self.api_base = self.api_base.rstrip("/")
            if self.api_base.endswith("/v1"):
                self.api_base = self.api_base[:-3]

        self.api_key = api_key
        self.temperature = temperature
        self.cache_db_path = Path(cache_db_path)
        self.last_error: Optional[str] = None
        self.interrupted: bool = False
        self.stats = {
            "live_llm": 0,
            "cache_hits": 0,
            "offline_fallbacks": 0,
            "latencies_seconds": [],
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "start_time": None,
            "total_duration_seconds": 0.0,
            "persona_counts": {},
        }
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initializes SQLite cache database."""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt TEXT,
                    response TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _get_cache(self, key: str) -> Optional[str]:
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM prompt_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _set_cache(self, key: str, prompt: str, response: str) -> None:
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_cache (cache_key, prompt, response, model) VALUES (?, ?, ?, ?)",
                (key, prompt, response, self.model),
            )

    def _format_error(self, e: Exception) -> str:
        """Extracts a human-readable, precise error explanation."""
        msg = str(e)
        if "ConnectionRefused" in msg or "Connection refused" in msg or "Failed to connect" in msg:
            return f"Connection refused to {self.api_base or 'endpoint'}. Is your local LLM server running?"
        if "404" in msg or "not found" in msg.lower():
            return f"Model '{self.model}' was not found at {self.api_base or 'endpoint'} (HTTP 404). Check 'ollama list' or model name."
        if "401" in msg or "Unauthorized" in msg or "invalid_api_key" in msg:
            return f"Authentication failed (HTTP 401). Invalid API key provided for provider '{self.provider}'."
        if "429" in msg or "rate_limit" in msg.lower():
            return f"Rate limit exceeded (HTTP 429) for provider '{self.provider}'."
        # Default clean single-line error
        return msg.strip().split("\n")[0]

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls configured LLM (Gemini, Claude, GPT, or local) via LiteLLM."""
        t0 = time.perf_counter()
        try:
            litellm.suppress_debug_info = True
            
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = litellm.completion(**kwargs)
            duration = time.perf_counter() - t0
            self.stats["latencies_seconds"].append(duration)

            content = response.choices[0].message.content or ""
            comp_tokens = 0
            prompt_tokens = 0
            if hasattr(response, "usage") and response.usage:
                comp_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            if comp_tokens == 0:
                comp_tokens = max(1, int(len(content) / 3.8))
            if prompt_tokens == 0:
                prompt_tokens = max(1, int((len(system_prompt) + len(user_prompt)) / 3.8))

            self.stats["completion_tokens"] += comp_tokens
            self.stats["prompt_tokens"] += prompt_tokens
            return content
        except Exception as e:
            self.last_error = self._format_error(e)
            return None

    def augment_scaffold_sync(
        self,
        scaffold: ScaffoldRecord,
        persona: Any,
        variations_count: int = 1,
    ) -> List[AugmentedRecord]:
        """Synchronously augments a scaffold for a specific persona."""
        results = []
        p_type = PersonaType(persona) if isinstance(persona, str) else persona
        cache_key = hashlib.sha256(f"{scaffold.id}_{p_type.value}_{self.model}".encode("utf-8")).hexdigest()

        # Check cache
        cached_resp = self._get_cache(cache_key)
        if cached_resp:
            self.stats["cache_hits"] += 1
            try:
                augmented_queries = json.loads(cached_resp)
            except Exception:
                augmented_queries = [cached_resp]
        else:
            system_prompt = PERSONA_SYSTEM_PROMPTS.get(
                p_type, "You are a domain expert asking precise questions about units of measurement."
            )
            user_prompt = (
                f"Paraphrase the following metrological question into {variations_count} distinct, realistic questions "
                f"from your persona's perspective:\n\n\"{scaffold.canonical_query}\"\n\n"
                f"Output only a JSON array of strings: [\"variation 1\", ...]"
            )

            llm_output = self._call_llm_api(system_prompt, user_prompt)
            if llm_output:
                self.stats["live_llm"] += 1
                try:
                    # Strip markdown code blocks if returned
                    clean_output = llm_output.strip()
                    if clean_output.startswith("```json"):
                        clean_output = clean_output[7:-3].strip()
                    elif clean_output.startswith("```"):
                        clean_output = clean_output[3:-3].strip()
                    augmented_queries = json.loads(clean_output)
                except Exception:
                    augmented_queries = [llm_output]
            else:
                self.stats["offline_fallbacks"] += 1
                # Deterministic fallback when no API key or offline
                if p_type == PersonaType.GENERAL_USER:
                    augmented_queries = [scaffold.canonical_query]
                else:
                    augmented_queries = [
                        f"[{p_type.value.replace('_', ' ').title()}] {scaffold.canonical_query}"
                    ]

            self._set_cache(cache_key, scaffold.canonical_query, json.dumps(augmented_queries))

        for idx, query in enumerate(augmented_queries):
            results.append(
                AugmentedRecord(
                    id=f"{scaffold.id}_{p_type.value}_{idx}",
                    scaffold_id=scaffold.id,
                    archetype=scaffold.archetype,
                    persona=p_type,
                    user_query=str(query),
                    ground_truth_answer=scaffold.ground_truth_answer,
                    entity_uri=scaffold.entity_uri,
                    quantity_kind_uri=scaffold.quantity_kind_uri,
                    temperature=self.temperature,
                    llm_generator=self.model,
                )
            )

        return results

    def augment_all(
        self,
        scaffolds: List[ScaffoldRecord],
        personas: Optional[List[PersonaType]] = None,
        variations_per_archetype: int = 2,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[AugmentedRecord]:
        """Augments a collection of scaffolds concurrently across all designated personas."""
        if personas is None or personas == "all" or (isinstance(personas, list) and "all" in personas):
            personas = list(PersonaType)
        else:
            personas = [PersonaType(p) if isinstance(p, str) else p for p in personas]

        t_start = time.perf_counter()
        work_items = [(scaffold, persona) for scaffold in scaffolds for persona in personas]
        augmented_records: List[AugmentedRecord] = []

        max_workers = min(self.concurrency_limit, max(1, len(work_items)))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                executor.submit(self.augment_scaffold_sync, scaffold, persona, variations_per_archetype)
                for scaffold, persona in work_items
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    augmented_records.extend(res)
                    for r in res:
                        p_val = r.persona.value if hasattr(r.persona, "value") else str(r.persona)
                        self.stats["persona_counts"][p_val] = self.stats["persona_counts"].get(p_val, 0) + 1
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(1)
        except KeyboardInterrupt:
            self.interrupted = True
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        self.stats["total_duration_seconds"] = time.perf_counter() - t_start
        return augmented_records

    def save_to_json(self, records: List[AugmentedRecord], output_path: str = "./data/augmented.json") -> Path:
        """Serializes augmented records to JSON with UTF-8 encoding for cross-platform compatibility."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            data = [r.model_dump() for r in records]
            json.dump(data, f, indent=2)
        return out
