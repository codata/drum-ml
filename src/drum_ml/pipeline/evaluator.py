import hashlib
import json
import os
import platform
import re
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from drum_ml.benchmark.models import (
    BenchmarkFormat,
    BenchmarkPerformanceMetrics,
    BenchmarkScorecard,
    EnvironmentProfile,
    TaskScore,
)
from drum_ml.symbolic.latex_parser import sanitize_latex_units
from drum_ml.symbolic.pint_engine import check_unit_conversion_equivalence


def detect_gpu_info() -> tuple[str | None, int | None, float | None]:
    """Detects available GPU/accelerator name, device count, and VRAM memory (in GB)."""
    # 1. Check PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            cnt = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            gpu_name = f"{cnt}x {name}" if cnt > 1 else name
            return gpu_name, cnt, vram
    except Exception:
        pass

    # 2. Check nvidia-smi CLI
    try:
        import subprocess

        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            cnt = len(lines)
            first_parts = lines[0].split(",")
            gpu_name = first_parts[0].strip()
            vram_mb = float(first_parts[1].strip()) if len(first_parts) > 1 else None
            vram_gb = round(vram_mb / 1024.0, 1) if vram_mb else None
            display_name = f"{cnt}x {gpu_name}" if cnt > 1 else gpu_name
            return display_name, cnt, vram_gb
    except Exception:
        pass

    # 3. Check macOS Apple Silicon / Metal Display
    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).decode()
            model = None
            cores = None
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    model = line.split(":", 1)[1].strip()
                elif "Total Number of Cores:" in line:
                    cores = line.split(":", 1)[1].strip()
            if model:
                gpu_str = f"{model} ({cores} GPU cores)" if cores else model
                return gpu_str, 1, None
        except Exception:
            pass

    return None, None, None


def classify_endpoint(
    endpoint: str | None = None,
    model_name: str = "",
    execution_type: str | None = None,
) -> tuple[str, str, str | None, bool]:
    """Classifies an inference endpoint into (execution_type, provider_name, sanitized_endpoint, is_local).

    Intelligently detects:
    1. Ground truth baselines.
    2. Explicit execution_type overrides ('local', 'cloud').
    3. Cloud-hosted models proxied through local daemons (e.g. model names with '-cloud', ':cloud', etc.).
    4. Local inference daemons (Ollama, LM Studio, vLLM, llama.cpp).
    5. Direct remote cloud provider APIs (OpenAI, Anthropic, Groq, Together, DeepSeek, OpenRouter, etc.).
    """
    if model_name in ["ground_truth_baseline", "baseline", "gold"]:
        return "baseline", "In-Memory Deterministic Baseline", None, True

    if not endpoint:
        return "local", "Local Inference", None, True

    ep = endpoint.strip()
    try:
        from urllib.parse import urlparse

        parsed = urlparse(ep)
        sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        host = (parsed.hostname or "").lower()
    except Exception:
        sanitized = ep
        host = ep.lower()

    model_lower = model_name.lower()
    has_cloud_tag = any(
        tag in model_lower for tag in ["-cloud", ":cloud", "/cloud", "_cloud", "cloud/"]
    ) or model_lower.endswith("cloud")

    local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "0:0:0:0:0:0:0:1"}
    is_local_host = host in local_hosts or "127.0.0.1" in ep or "localhost" in ep

    # Explicit override handling
    if execution_type and execution_type.lower() in ["cloud", "remote"]:
        provider = (
            "Ollama Cloud (Proxy via Local Host)"
            if (is_local_host and "11434" in ep)
            else (f"Cloud API ({host or model_name})")
        )
        return "cloud", provider, sanitized, False

    if execution_type and execution_type.lower() in ["local"]:
        provider = "Ollama (Local Host)" if "11434" in ep else "Local Server"
        return "local", provider, sanitized, True

    # Automatic classification
    if is_local_host:
        # Check if local daemon is running a cloud-hosted / offloaded model
        if has_cloud_tag:
            if "11434" in ep:
                provider = "Ollama Cloud (Proxy via Local Host)"
            elif "1234" in ep:
                provider = "LM Studio Cloud Proxy"
            else:
                provider = "Local Proxy (Cloud-Hosted Model)"
            return "cloud", provider, sanitized, False

        if "11434" in ep:
            provider = "Ollama (Local Host)"
        elif "1234" in ep:
            provider = "LM Studio (Local Host)"
        elif "8000" in ep or "vllm" in ep.lower():
            provider = "vLLM (Local Host)"
        elif "8080" in ep or "llama" in ep.lower():
            provider = "llama.cpp (Local Host)"
        else:
            provider = "Local OpenAI-Compatible Server"
        return "local", provider, sanitized, True

    # Cloud endpoints
    if "openai.com" in host:
        provider = "OpenAI API (Cloud)"
    elif "anthropic.com" in host:
        provider = "Anthropic API (Cloud)"
    elif "groq.com" in host:
        provider = "Groq API (Cloud)"
    elif "together.xyz" in host or "together.ai" in host:
        provider = "Together AI (Cloud)"
    elif "deepseek.com" in host:
        provider = "DeepSeek API (Cloud)"
    elif "openrouter.ai" in host:
        provider = "OpenRouter (Cloud)"
    elif "mistral.ai" in host:
        provider = "Mistral AI (Cloud)"
    elif "cohere.com" in host or "cohere.ai" in host:
        provider = "Cohere (Cloud)"
    elif "azure.com" in host:
        provider = "Azure OpenAI (Cloud)"
    elif "bedrock" in host or "amazonaws.com" in host:
        provider = "AWS Bedrock (Cloud)"
    else:
        provider = f"Remote Cloud Endpoint ({host or 'unknown'})"

    return "cloud", provider, sanitized, False


def get_anonymous_environment(
    endpoint: str | None = None,
    model_name: str = "",
    execution_type: str | None = None,
) -> EnvironmentProfile:
    """Collects an anonymized system, hardware, GPU accelerator, and model execution profile."""
    os_sys = platform.system()
    os_rel = platform.release()
    os_str = f"{os_sys} {os_rel}"
    if os_sys == "Darwin":
        mac_v = platform.mac_ver()[0]
        if mac_v:
            os_str = f"macOS {mac_v} (Darwin {os_rel})"

    arch = platform.machine() or platform.processor() or "unknown"
    cpu_count = os.cpu_count() or 1

    mem_gb: float | None = None
    try:
        if os_sys == "Darwin":
            import subprocess

            out = (
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2).decode().strip()
            )
            mem_gb = round(int(out) / (1024**3), 1)
        elif os_sys == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = round(mem_kb / (1024**2), 1)
                        break
    except Exception:
        pass

    gpu_name, gpu_cnt, gpu_vram = detect_gpu_info()
    exec_type, provider, sanitized_ep, is_local = classify_endpoint(
        endpoint=endpoint, model_name=model_name, execution_type=execution_type
    )

    return EnvironmentProfile(
        execution_type=exec_type,
        provider=provider,
        endpoint=sanitized_ep,
        is_local_inference=is_local,
        os=os_str,
        architecture=arch,
        cpu_count=cpu_count,
        total_memory_gb=mem_gb,
        gpu=gpu_name,
        gpu_count=gpu_cnt,
        gpu_memory_gb=gpu_vram,
        python_version=platform.python_version(),
        platform=platform.platform(terse=True),
    )


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes the SHA-256 hash of a benchmark dataset file for integrity and versioning."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return "sha256:unknown"
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


class ModelQueryResult(BaseModel):
    """Result of querying a model endpoint including token usage and latency."""

    content: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    reasoning_content: str | None = None

    def __str__(self) -> str:
        return self.content


class MEvalBenchmark:
    """Evaluator and grader for the DRUM Metrology Benchmark suite."""

    def __init__(self, benchmark_file: str = "./dataset/drum_benchmark_mcq.jsonl"):
        self.benchmark_file = Path(benchmark_file)

    @staticmethod
    def format_prompt(record: dict[str, Any]) -> str:
        """Format a benchmark record into prompt text for LLM inference."""
        question = record.get("question", "").strip()
        options = record.get("options", [])
        if options:
            formatted_opts = []
            for opt in options:
                k = opt.get("key", "")
                t = opt.get("text", "").strip()
                formatted_opts.append(f"{k}. {t}")
            opts_str = "\n".join(formatted_opts)
            return (
                f"Question: {question}\n\n"
                f"Options:\n{opts_str}\n\n"
                f"Instructions: Analyze the options carefully and select the single correct letter (A, B, C, or D).\n"
                f'Respond in valid JSON format with keys "answer" (the single uppercase letter) and "explanation" (brief justification).\n\n'
                f"Example response format:\n"
                f"```json\n"
                f'{{\n  "answer": "A",\n  "explanation": "Brief explanation of the metrological rationale."\n}}\n'
                f"```"
            )
        return (
            f"Question: {question}\n\n"
            f"Instructions: Provide the exact numerical value and unit for the answer.\n"
            f'Respond in valid JSON format with keys "answer" (the exact value and unit) and "explanation" (brief justification).\n\n'
            f"Example response format:\n"
            f"```json\n"
            f'{{\n  "answer": "1.054571817e-34 J s",\n  "explanation": "Exact by the 2019 SI definition."\n}}\n'
            f"```"
        )

    @staticmethod
    def query_model_api(
        endpoint: str,
        model_name: str,
        prompt: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int | None = None,
    ) -> ModelQueryResult:
        """Queries an LLM via LiteLLM or an OpenAI-compatible endpoint with telemetry."""
        start_t = time.perf_counter()

        def make_result(
            content: str,
            prompt_toks: int = 0,
            completion_toks: int = 0,
            reasoning: str | None = None,
        ) -> ModelQueryResult:
            duration = max(time.perf_counter() - start_t, 1e-4)
            if prompt_toks <= 0 and prompt:
                prompt_toks = max(1, len(prompt) // 4)
            if completion_toks <= 0 and content:
                completion_toks = max(1, len(content) // 4)
            tot_toks = prompt_toks + completion_toks
            tok_per_sec = (
                (completion_toks / duration) if duration > 0 and completion_toks > 0 else 0.0
            )

            return ModelQueryResult(
                content=content,
                latency_seconds=round(duration, 3),
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=tot_toks,
                tokens_per_second=round(tok_per_sec, 2),
                reasoning_content=reasoning,
            )

        import os

        # Determine the appropriate API key based on endpoint and model if api_key is not explicitly given
        effective_api_key = api_key
        if not effective_api_key:
            endpoint_lower = (endpoint or "").lower()
            model_lower = (model_name or "").lower()
            if "openrouter" in endpoint_lower or "openrouter" in model_lower:
                effective_api_key = (
                    os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("OPENROUTER_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                )
            elif "anthropic" in endpoint_lower or "claude" in model_lower:
                effective_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
                    "OPENAI_API_KEY"
                )
            elif "groq" in endpoint_lower:
                effective_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get(
                    "OPENAI_API_KEY"
                )
            elif "together" in endpoint_lower:
                effective_api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get(
                    "OPENAI_API_KEY"
                )
            elif "deepseek" in endpoint_lower:
                effective_api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
                    "OPENAI_API_KEY"
                )
            elif "openai" in endpoint_lower:
                effective_api_key = os.environ.get("OPENAI_API_KEY")
            else:
                effective_api_key = (
                    os.environ.get("OPENROUTER_API_KEY")
                    if "openrouter" in endpoint_lower
                    else (
                        os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("OPENROUTER_API_KEY")
                        or os.environ.get("ANTHROPIC_API_KEY")
                        or os.environ.get("GEMINI_API_KEY")
                    )
                )

        # 1. Try LiteLLM first for robust multi-provider handling (Ollama, OpenAI, Claude, vLLM, OpenRouter)
        try:
            import litellm

            litellm.suppress_debug_info = True

            model_target = model_name
            api_base = endpoint.rstrip("/")
            if "11434" in endpoint and not model_target.startswith("ollama"):
                model_target = f"ollama/{model_name}"
                api_base = "http://localhost:11434"
            elif "openrouter.ai" in endpoint and not model_target.startswith("openrouter/"):
                model_target = f"openrouter/{model_name}"
                api_base = "https://openrouter.ai/api/v1"

            call_kwargs: dict[str, Any] = {
                "model": model_target,
                "api_base": api_base,
                "api_key": effective_api_key or "sk-local",
                "extra_headers": {
                    "HTTP-Referer": "https://drum.codata.org",
                    "X-Title": "DRUM-ML Metrology Benchmark",
                },
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert metrologist and physicist. Answer with extreme precision.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "timeout": timeout,
            }
            if max_tokens and max_tokens > 0:
                if any(m in model_target.lower() for m in ["o1", "o3", "reasoner"]):
                    call_kwargs["max_completion_tokens"] = max_tokens
                else:
                    call_kwargs["max_tokens"] = max_tokens

            resp = litellm.completion(**call_kwargs)
            choices = resp.choices if hasattr(resp, "choices") else []
            if choices:
                msg = choices[0].message
                content = getattr(msg, "content", "") or ""
                reasoning = getattr(msg, "reasoning_content", None) or getattr(
                    msg, "thinking_content", None
                )
                final_content = str(content).strip()
                if not final_content and reasoning:
                    final_content = str(reasoning).strip()
                elif reasoning and reasoning not in final_content:
                    final_content = f"<think>\n{reasoning}\n</think>\n{final_content}".strip()

                usage = getattr(resp, "usage", None)
                p_toks = getattr(usage, "prompt_tokens", 0) if usage else 0
                c_toks = getattr(usage, "completion_tokens", 0) if usage else 0

                return make_result(
                    final_content, p_toks, c_toks, reasoning=str(reasoning) if reasoning else None
                )
        except Exception:
            pass

        # 2. Fallback to direct HTTP request with retry on 429/503
        import urllib.error
        import urllib.request

        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions") and not url.endswith("/completions"):
            url = f"{url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://drum.codata.org",
            "X-Title": "DRUM-ML Metrology Benchmark",
        }
        if effective_api_key:
            headers["Authorization"] = f"Bearer {effective_api_key}"

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert metrologist and physicist. Answer metrological questions with extreme precision.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        max_attempts = 3
        for attempt in range(max_attempts):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "") or ""
                        reasoning = msg.get("reasoning_content") or msg.get("thinking_content")
                        final_content = str(content).strip()
                        if not final_content and reasoning:
                            final_content = str(reasoning).strip()
                        elif reasoning and reasoning not in final_content:
                            final_content = (
                                f"<think>\n{reasoning}\n</think>\n{final_content}".strip()
                            )

                        usage = data.get("usage", {})
                        p_toks = usage.get("prompt_tokens", 0)
                        c_toks = usage.get("completion_tokens", 0)

                        return make_result(
                            final_content,
                            p_toks,
                            c_toks,
                            reasoning=str(reasoning) if reasoning else None,
                        )
                    if "error" in data:
                        err_val = data["error"]
                        err_text = (
                            err_val.get("message", str(err_val))
                            if isinstance(err_val, dict)
                            else str(err_val)
                        )
                        return make_result(f"ERROR_CALLING_MODEL: {err_text}")
                    return make_result("")
            except urllib.error.HTTPError as e:
                err_detail = str(e.reason)
                try:
                    err_body = e.read().decode("utf-8")
                    err_json = json.loads(err_body)
                    if "error" in err_json:
                        e_obj = err_json["error"]
                        err_detail = (
                            e_obj.get("message", str(e_obj))
                            if isinstance(e_obj, dict)
                            else str(e_obj)
                        )
                except Exception:
                    pass

                # If rate limited (429) or overloaded (503), retry with brief delay
                if e.code in (429, 503) and attempt < max_attempts - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue

                return make_result(f"ERROR_CALLING_MODEL: HTTP Error {e.code}: {err_detail}")
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    continue
                return make_result(f"ERROR_CALLING_MODEL: {e}")

        return make_result("ERROR_CALLING_MODEL: Maximum retry attempts exceeded")

    @staticmethod
    def check_fatal_inference_error(error_str: str) -> str | None:
        """Checks if an inference error is a fatal configuration, authentication, or connection error that should halt the evaluation."""
        if not error_str or not str(error_str).startswith("ERROR_CALLING_MODEL:"):
            return None
        err = str(error_str).lower()
        if (
            "401" in err
            or "unauthorized" in err
            or "invalid_api_key" in err
            or "authentication" in err
        ):
            return "Authentication Failed (HTTP 401): Invalid or missing API key. Please check your API key configuration."
        if "403" in err or "forbidden" in err or "permission_denied" in err:
            return "Access Forbidden (HTTP 403): You do not have permission to access this model or endpoint."
        if "404" in err or "not found" in err:
            return "Model / Endpoint Not Found (HTTP 404): The requested model or endpoint URL does not exist."
        if "connection refused" in err or "failed to connect" in err:
            return "Connection Refused: Unable to connect to inference server. Is the local daemon running?"
        return None

    def load_benchmark_records(self) -> list[dict[str, Any]]:
        """Load benchmark samples from JSONL."""
        records = []
        if not self.benchmark_file.exists():
            return records
        with open(self.benchmark_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def get_checkpoint_path(output_path: Path | str) -> Path:
        """Returns the canonical checkpoint file path for a target output scorecard path."""
        p = Path(output_path)
        return p.parent / f".{p.stem}.checkpoint.json"

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path | str, expected_model_name: str | None = None
    ) -> tuple[dict[str, dict[str, Any]], float]:
        """Loads completed results from an evaluation checkpoint file.

        Returns (results_map_by_id, accumulated_duration_seconds).
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            return {}, 0.0

        try:
            with open(ckpt, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return {}, 0.0

            if expected_model_name and data.get("model_name") != expected_model_name:
                return {}, 0.0

            raw_results = data.get("results", [])
            results_map = {r["id"]: r for r in raw_results if isinstance(r, dict) and "id" in r}
            accumulated_time = float(data.get("accumulated_duration_seconds", 0.0))
            return results_map, accumulated_time
        except Exception:
            return {}, 0.0

    @staticmethod
    def save_checkpoint(
        checkpoint_path: Path | str,
        model_name: str,
        benchmark_file: Path | str,
        total_records: int,
        results_list: list[dict[str, Any]],
        accumulated_duration_seconds: float,
    ) -> Path:
        """Atomically saves evaluation progress to a checkpoint file."""
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")

        payload = {
            "model_name": model_name,
            "benchmark_file": str(benchmark_file),
            "total_records": total_records,
            "completed_count": len(results_list),
            "accumulated_duration_seconds": accumulated_duration_seconds,
            "results": results_list,
        }

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        tmp.replace(ckpt)
        return ckpt

    @staticmethod
    def remove_checkpoint(checkpoint_path: Path | str) -> bool:
        """Removes the checkpoint file once evaluation completes."""
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            try:
                ckpt.unlink()
                return True
            except Exception:
                pass
        return False

    @staticmethod
    def parse_mcq_response(response_text: str) -> tuple[str | None, str | None]:
        """Extracts (predicted_key, explanation) from an LLM response.

        Attempts multi-tier parsing:
        1. JSON parsing (fenced ```json ... ``` blocks and raw JSON objects).
        2. Thought-tag stripping (<think>...</think>).
        3. Heuristic regex pattern matching for non-compliant models.
        """
        text = response_text.strip()
        if not text:
            return None, None

        # Strip reasoning tags if present
        cleaned_text = text
        if "</think>" in cleaned_text:
            cleaned_text = cleaned_text.split("</think>")[-1].strip()
        elif "<think>" in cleaned_text:
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_text).strip()

        # Helper to extract letter from value
        def extract_letter_from_val(val: Any) -> str | None:
            if not isinstance(val, str):
                return None
            v = val.strip()
            if v.upper() in ["A", "B", "C", "D"]:
                return v.upper()
            m = re.search(r"\b([A-Da-d])\b", v)
            if m:
                return m.group(1).upper()
            return None

        # Tier 1: Try JSON extraction
        # 1a. Markdown fenced blocks
        json_candidates = re.findall(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_text, flags=re.IGNORECASE
        )
        # 1b. Outermost/balanced JSON objects
        raw_objs = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_text)
        json_candidates.extend(raw_objs)

        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            json_candidates.insert(0, cleaned_text)

        key_aliases = [
            "answer",
            "option",
            "choice",
            "selected_option",
            "correct_option",
            "key",
            "selected",
            "response",
        ]
        exp_aliases = [
            "explanation",
            "reasoning",
            "rationale",
            "justification",
            "details",
            "notes",
        ]

        for cand in json_candidates:
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    pred_key = None
                    for k in key_aliases:
                        if k in data:
                            pred_key = extract_letter_from_val(data[k])
                            if pred_key:
                                break

                    pred_exp = None
                    for e in exp_aliases:
                        if e in data and isinstance(data[e], str):
                            pred_exp = data[e].strip()
                            break

                    if pred_key:
                        return pred_key, pred_exp
            except Exception:
                continue

        # Tier 2: Standalone Letter
        if cleaned_text.upper() in ["A", "B", "C", "D"]:
            return cleaned_text.upper(), None

        # Tier 3: Regex heuristics
        patterns = [
            r"(?:the\s+correct\s+answer\s+is|correct\s+option\s+is|answer\s*[:=]?)\s*[\*\_]*\(?([A-Da-d])\)?",
            r'"answer"\s*:\s*"([A-Da-d])"',
            r'"option"\s*:\s*"([A-Da-d])"',
            r"[\*\_]*\(([A-Da-d])\)[\*\_]*",
            r"[\*\_]*\[([A-Da-d])\][\*\_]*",
            r"option\s+([A-Da-d])\b",
            r"^\s*([A-Da-d])[\.\:\)]\s*",
            r"\b([A-Da-d])\b",
        ]
        for pat in patterns:
            m = re.search(pat, cleaned_text, re.IGNORECASE)
            if m:
                return m.group(1).upper(), None

        return None, None

    @staticmethod
    def parse_freeform_response(response_text: str) -> tuple[str, str | None]:
        """Extracts candidate numerical quantity and unit from an LLM free-form response.

        Attempts:
        1. JSON parsing for keys 'answer', 'value', 'result'.
        2. Boxed expression \\boxed{...} extraction.
        3. LaTeX equation / rhs isolation.
        4. Stripping conversational filler.
        """
        text = response_text.strip()
        if not text:
            return "", None

        # Strip reasoning tags (<think>...</think>)
        cleaned_text = text
        if "</think>" in cleaned_text:
            cleaned_text = cleaned_text.split("</think>")[-1].strip()
        elif "<think>" in cleaned_text:
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_text).strip()

        # Tier 1: JSON extraction
        json_candidates = re.findall(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_text, flags=re.IGNORECASE
        )
        if not json_candidates:
            raw_match = re.search(r"(\{[\s\S]*\})", cleaned_text)
            if raw_match:
                json_candidates = [raw_match.group(1)]

        for cand in json_candidates:
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    ans_val = data.get("answer") or data.get("value") or data.get("result")
                    exp_val = data.get("explanation") or data.get("reasoning")
                    if ans_val and isinstance(ans_val, (str, int, float)):
                        return str(ans_val).strip(), str(exp_val).strip() if exp_val else None
            except Exception:
                continue

        # Tier 2: Boxed expression \boxed{...} (with nested brace support)
        idx = cleaned_text.find(r"\boxed{")
        if idx != -1:
            start = idx + len(r"\boxed{")
            depth = 1
            i = start
            while i < len(cleaned_text) and depth > 0:
                if cleaned_text[i] == "{":
                    depth += 1
                elif cleaned_text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                return cleaned_text[start : i - 1].strip(), None

        # Tier 3: Equation right-hand side (e.g. \hbar = 1.05457... J s)
        m_eq = re.search(
            r"(?:=|is\s+defined\s+as|is\s+equal\s+to|is\s+exactly|is)\s*[:=]?\s*([0-9\.\-e\+\^\*\/\s\\a-zA-Z]+(?:\s*[a-zA-Z\^\-\/]+)?)",
            cleaned_text,
            re.IGNORECASE,
        )
        if m_eq:
            cand = m_eq.group(1).strip().rstrip(".")
            if len(cand.split()) <= 6 and not cand.startswith(r"\boxed"):
                return cand, None

        # Tier 4: Fallback
        return cleaned_text, None

    def extract_mcq_answer(self, response_text: str) -> str | None:
        """Extracts the predicted MCQ option letter (A, B, C, D) from an LLM response."""
        pred_key, _ = self.parse_mcq_response(response_text)
        return pred_key

    def grade_response(
        self,
        record: dict[str, Any],
        predicted_text: str,
    ) -> dict[str, Any]:
        """Grades a response against ground truth using MCQ extraction or symbolic equivalence."""
        fmt = record.get("format", BenchmarkFormat.MCQ.value)
        correct_key = record.get("correct_option_key")
        gt_answer = record.get("ground_truth_answer", "")

        # Case 1: Standard Chat JSONL compatibility (fallback)
        if not correct_key and "messages" in record:
            messages = record.get("messages", [])
            gt_answer = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
            fmt = BenchmarkFormat.FREE_FORM.value

        # Case 0: Explicit Model Calling / Network Error
        if predicted_text.startswith("ERROR_CALLING_MODEL:"):
            return {
                "passed": False,
                "eval_status": "error",
                "error_type": "provider_error",
                "format": fmt,
                "predicted_key": None,
                "expected_key": correct_key,
                "predicted_raw": predicted_text,
                "error": predicted_text,
            }

        # Track A: MCQ Grading
        if fmt == BenchmarkFormat.MCQ.value or correct_key:
            pred_key, pred_exp = self.parse_mcq_response(predicted_text)
            passed = (pred_key == correct_key) if (pred_key and correct_key) else False
            if passed:
                eval_status = "passed"
                err_msg = None
                error_type = None
            elif not pred_key:
                eval_status = "error"
                err_msg = "Failed to extract MCQ choice letter from model output"
                error_type = "unparseable_response"
            else:
                eval_status = "incorrect"
                err_msg = f"Expected option {correct_key}, got {pred_key}"
                error_type = "mcq_choice_mismatch"

            grade_dict: dict[str, Any] = {
                "passed": passed,
                "eval_status": eval_status,
                "error_type": error_type,
                "format": BenchmarkFormat.MCQ.value,
                "predicted_key": pred_key,
                "expected_key": correct_key,
                "predicted_raw": predicted_text,
                "error": err_msg,
            }
            if pred_exp:
                grade_dict["predicted_explanation"] = pred_exp
            return grade_dict

        # Track B: Free-Form / Symbolic Physics Equivalence
        pred_ans, pred_exp = self.parse_freeform_response(predicted_text)
        gt_clean = sanitize_latex_units(gt_answer)
        pred_clean = sanitize_latex_units(pred_ans if pred_ans else predicted_text)

        exact_match = (gt_clean.lower() == pred_clean.lower()) or (
            gt_answer.strip().lower() == (pred_ans or predicted_text).strip().lower()
        )
        is_sym_eq, sym_err = False, None

        if not exact_match and pred_clean:
            is_sym_eq, sym_err = check_unit_conversion_equivalence(gt_clean, pred_clean)

        passed = exact_match or is_sym_eq
        if passed:
            eval_status = "passed"
            err_msg = None
            error_type = None
        elif sym_err and (
            "Pint evaluation error" in sym_err or "SymPy evaluation error" in sym_err
        ):
            eval_status = "error"
            err_msg = sym_err
            error_type = "processing_error"
        elif sym_err and "Dimensional incompatibility" in sym_err:
            eval_status = "incorrect"
            err_msg = sym_err
            error_type = "dimensional_mismatch"
        elif sym_err and "Magnitude mismatch" in sym_err:
            eval_status = "incorrect"
            err_msg = sym_err
            error_type = "magnitude_mismatch"
        elif not pred_ans and not predicted_text.strip():
            eval_status = "error"
            err_msg = "Model returned empty response"
            error_type = "empty_response"
        else:
            eval_status = "incorrect"
            err_msg = sym_err or f"Expected '{gt_answer}', got '{pred_ans or predicted_text}'"
            error_type = "value_mismatch"

        grade_dict = {
            "passed": passed,
            "eval_status": eval_status,
            "error_type": error_type,
            "format": BenchmarkFormat.FREE_FORM.value,
            "predicted_raw": predicted_text,
            "extracted_answer": pred_ans or predicted_text,
            "exact_match": exact_match,
            "symbolic_match": is_sym_eq,
            "error": err_msg,
        }
        if pred_exp:
            grade_dict["predicted_explanation"] = pred_exp
        return grade_dict

    def compute_scorecard(
        self,
        model_name: str,
        results: list[dict[str, Any]],
        benchmark_version: str | None = None,
        total_duration_seconds: float = 0.0,
        environment: EnvironmentProfile | dict[str, Any] | None = None,
        endpoint: str | None = None,
        execution_type: str | None = None,
    ) -> BenchmarkScorecard:
        """Computes stratified scores across all 6 tasks, formats, difficulties, token telemetry, and environment profile."""
        total = len(results)
        passed = sum(1 for r in results if r["grade"]["passed"])
        overall_pct = (passed / total * 100.0) if total > 0 else 0.0

        # Sub-breakdowns
        task_stats: dict[str, dict[str, int]] = {}
        fmt_stats: dict[str, dict[str, int]] = {}
        diff_stats: dict[str, dict[str, int]] = {}

        # Performance Telemetry
        latencies: list[float] = []
        prompt_toks = 0
        comp_toks = 0

        for r in results:
            t = r.get("task", "unknown")
            f = r.get("format", "unknown")
            d = r.get("difficulty", "intermediate")
            is_p = 1 if r["grade"]["passed"] else 0

            # Task
            if t not in task_stats:
                task_stats[t] = {"total": 0, "passed": 0}
            task_stats[t]["total"] += 1
            task_stats[t]["passed"] += is_p

            # Format
            if f not in fmt_stats:
                fmt_stats[f] = {"total": 0, "passed": 0}
            fmt_stats[f]["total"] += 1
            fmt_stats[f]["passed"] += is_p

            # Difficulty
            if d not in diff_stats:
                diff_stats[d] = {"total": 0, "passed": 0}
            diff_stats[d]["total"] += 1
            diff_stats[d]["passed"] += is_p

            # Metrics
            m = r.get("metrics")
            if isinstance(m, dict):
                lat = m.get("latency_seconds")
                if isinstance(lat, (int, float)) and lat > 0:
                    latencies.append(float(lat))
                prompt_toks += int(m.get("prompt_tokens") or 0)
                comp_toks += int(m.get("completion_tokens") or 0)

        tot_toks = prompt_toks + comp_toks
        dur = total_duration_seconds if total_duration_seconds > 0 else sum(latencies)
        avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
        sorted_lat = sorted(latencies)
        p50 = statistics.median(sorted_lat) if sorted_lat else 0.0
        p95 = (
            sorted_lat[int(len(sorted_lat) * 0.95)]
            if len(sorted_lat) >= 20
            else (sorted_lat[-1] if sorted_lat else 0.0)
        )
        avg_p_toks = (prompt_toks / total) if total > 0 else 0.0
        avg_c_toks = (comp_toks / total) if total > 0 else 0.0
        sum_inference_time = sum(latencies) if latencies else dur
        avg_tps = (
            (comp_toks / sum_inference_time) if (sum_inference_time > 0 and comp_toks > 0) else 0.0
        )

        perf_metrics = BenchmarkPerformanceMetrics(
            total_duration_seconds=round(dur, 2),
            avg_latency_seconds=round(avg_lat, 3),
            p50_latency_seconds=round(p50, 3),
            p95_latency_seconds=round(p95, 3),
            total_prompt_tokens=prompt_toks,
            total_completion_tokens=comp_toks,
            total_tokens=tot_toks,
            avg_prompt_tokens=round(avg_p_toks, 1),
            avg_completion_tokens=round(avg_c_toks, 1),
            avg_tokens_per_second=round(avg_tps, 2),
        )

        env_prof = None
        if isinstance(environment, EnvironmentProfile):
            env_prof = environment
        elif isinstance(environment, dict):
            env_prof = EnvironmentProfile(**environment)
        else:
            env_prof = get_anonymous_environment(
                endpoint=endpoint, model_name=model_name, execution_type=execution_type
            )

        ver = benchmark_version or compute_file_sha256(self.benchmark_file)

        def to_task_score_dict(d_dict: dict[str, dict[str, int]]) -> dict[str, TaskScore]:
            out = {}
            for k, v in d_dict.items():
                pct = (v["passed"] / v["total"] * 100.0) if v["total"] > 0 else 0.0
                out[k] = TaskScore(task=k, total=v["total"], passed=v["passed"], accuracy_pct=pct)
            return out

        return BenchmarkScorecard(
            model_name=model_name,
            benchmark_version=ver,
            benchmark_file=str(self.benchmark_file),
            total_samples=total,
            passed_samples=passed,
            overall_accuracy_pct=overall_pct,
            task_breakdown=to_task_score_dict(task_stats),
            format_breakdown=to_task_score_dict(fmt_stats),
            difficulty_breakdown=to_task_score_dict(diff_stats),
            environment=env_prof,
            metrics=perf_metrics,
            detailed_results=results,
        )

    def print_scorecard(
        self, scorecard: BenchmarkScorecard, console: Console | None = None
    ) -> None:
        """Prints a rich, formatted evaluation scorecard to the terminal."""
        con = console or Console()

        # 1. Benchmark & Environment Metadata Table
        meta_table = Table(
            title=f"DRUM Metrology Benchmark: {scorecard.model_name}",
            title_style="bold magenta",
            header_style="bold cyan",
            show_header=False,
        )
        meta_table.add_column("Property", style="bold")
        meta_table.add_column("Value", style="cyan")

        meta_table.add_row("Model Name", scorecard.model_name)
        meta_table.add_row("Evaluation Timestamp (UTC)", scorecard.timestamp)
        ver_display = scorecard.benchmark_version
        if len(ver_display) > 28:
            ver_display = ver_display[:28] + "..."
        meta_table.add_row("Benchmark Version", ver_display)
        if scorecard.benchmark_file:
            meta_table.add_row("Benchmark Dataset File", scorecard.benchmark_file)

        if scorecard.environment:
            env = scorecard.environment
            if env.execution_type == "local":
                meta_table.add_row(
                    "Execution Target", f"💻 Local Inference Engine ({env.provider})"
                )
                if env.endpoint:
                    meta_table.add_row("Local Endpoint", env.endpoint)
                host_str = f"{env.os} | {env.architecture}"
                if env.cpu_count:
                    host_str += f" ({env.cpu_count} CPUs"
                    if env.total_memory_gb:
                        host_str += f", {env.total_memory_gb} GB RAM"
                    host_str += ")"
                meta_table.add_row("Host Hardware", host_str)
                if env.gpu:
                    gpu_display = env.gpu
                    if env.gpu_memory_gb:
                        gpu_display += f" ({env.gpu_memory_gb} GB VRAM)"
                    meta_table.add_row("Inference GPU", gpu_display)
            elif env.execution_type == "cloud":
                meta_table.add_row("Execution Target", f"☁️ Remote Cloud API ({env.provider})")
                if env.endpoint:
                    meta_table.add_row("Cloud Endpoint", env.endpoint)
                client_str = f"{env.os} | {env.architecture} (Client Runner)"
                meta_table.add_row("Client Runner Host", client_str)
            else:
                meta_table.add_row("Execution Target", f"🎯 {env.provider}")

            meta_table.add_row("Python Runtime", f"Python {env.python_version}")

        con.print("\n")
        con.print(meta_table)

        # 2. Performance & Telemetry Table (if metrics present)
        if scorecard.metrics and (
            scorecard.metrics.total_tokens > 0 or scorecard.metrics.total_duration_seconds > 0
        ):
            m = scorecard.metrics
            perf_table = Table(
                title="⚡ Runtime & Inference Throughput Telemetry",
                title_style="bold yellow",
                header_style="bold cyan",
            )
            perf_table.add_column("Metric", style="bold")
            perf_table.add_column("Value", justify="right")

            mins = int(m.total_duration_seconds // 60)
            secs = m.total_duration_seconds % 60
            dur_str = (
                f"{mins}m {secs:.1f}s ({m.total_duration_seconds:.2f}s)"
                if mins > 0
                else f"{m.total_duration_seconds:.2f}s"
            )

            perf_table.add_row("Total Evaluation Duration", dur_str)
            perf_table.add_row(
                "Average Latency per Sample",
                f"{m.avg_latency_seconds:.3f}s (P50: {m.p50_latency_seconds:.3f}s, P95: {m.p95_latency_seconds:.3f}s)",
            )
            perf_table.add_row(
                "Total Tokens Processed",
                f"{m.total_tokens:,} (Prompt: {m.total_prompt_tokens:,}, Output: {m.total_completion_tokens:,})",
            )
            perf_table.add_row(
                "Avg Output Tokens per Sample", f"{m.avg_completion_tokens:.1f} tokens"
            )
            perf_table.add_row(
                "Generation Speed (Throughput)",
                f"[bold green]{m.avg_tokens_per_second:.2f} tokens/sec[/bold green]"
                if m.avg_tokens_per_second > 0
                else "N/A",
            )
            con.print("\n")
            con.print(perf_table)

        # 3. Metrology Accuracy by Category Table
        table = Table(
            title=f"🎯 Metrological Accuracy by Sub-Discipline: {scorecard.model_name}",
            title_style="bold magenta",
            header_style="bold cyan",
        )
        table.add_column("Category / Sub-Discipline", style="bold")
        table.add_column("Samples", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Accuracy", justify="right")

        task_names = {
            "constants": "1. Fundamental Constants & SI 2019",
            "dimensions": "2. Dimensional Decomposition & Base SI",
            "conversions": "3. Unit Conversions & Affine Offsets",
            "homogeneity": "4. Error Detection & Homogeneity",
            "conventions": "5. SI Typography & Metrological Rules",
            "uncertainty": "6. Metrological Uncertainty (GUM)",
        }

        for task_key, task_label in task_names.items():
            if task_key in scorecard.task_breakdown:
                ts = scorecard.task_breakdown[task_key]
                color = (
                    "green"
                    if ts.accuracy_pct >= 80
                    else "yellow"
                    if ts.accuracy_pct >= 50
                    else "red"
                )
                table.add_row(
                    task_label,
                    str(ts.total),
                    str(ts.passed),
                    f"[{color}]{ts.accuracy_pct:.1f}%[/{color}]",
                )

        table.add_section()
        ov_color = (
            "bold green"
            if scorecard.overall_accuracy_pct >= 80
            else "bold yellow"
            if scorecard.overall_accuracy_pct >= 50
            else "bold red"
        )
        table.add_row(
            "OVERALL DRUM BENCHMARK SCORE",
            str(scorecard.total_samples),
            str(scorecard.passed_samples),
            f"[{ov_color}]{scorecard.overall_accuracy_pct:.2f}%[/{ov_color}]",
        )

        con.print("\n")
        con.print(table)
        con.print("\n")
