"""Multi-provider LLM client with fallback chain.

Uses LiteLLM for provider abstraction. Supports retry logic,
cost tracking, rate limiting, and automatic fallback to backup models.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque

import litellm

logger = logging.getLogger(__name__)

# Suppress noisy LiteLLM logging
litellm.set_verbose = False
for _name in (
    "LiteLLM", "litellm", "LiteLLM Proxy", "LiteLLM Router",
    "openai", "openai._base_client", "httpx", "httpcore",
):
    logging.getLogger(_name).setLevel(logging.WARNING)


class _RateLimiter:
    """Sliding-window rate limiter for LLM API calls."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.window = 60.0
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _purge(self, now: float):
        while self.timestamps and now - self.timestamps[0] > self.window:
            self.timestamps.popleft()

    def wait_sync(self):
        """Block until we can make a call without exceeding RPM."""
        while True:
            now = time.monotonic()
            self._purge(now)
            if len(self.timestamps) < self.rpm:
                self.timestamps.append(now)
                return
            sleep_time = self.timestamps[0] + self.window - now + 0.1
            time.sleep(max(0, sleep_time))

    async def wait_async(self):
        """Async version of rate limiter."""
        async with self._lock:
            while True:
                now = time.monotonic()
                self._purge(now)
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                sleep_time = self.timestamps[0] + self.window - now + 0.1
                await asyncio.sleep(max(0, sleep_time))


class LLMClient:
    """LLM client with fallback chain, retry logic, and cost tracking."""

    def __init__(
        self,
        model: str,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        rate_limit_retries: int = 8,
        rate_limit_base_wait: float = 5.0,
        rpm: int = 40,
        timeout: int = 120,
        system_message: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_base_wait = rate_limit_base_wait
        self.timeout = timeout
        self.system_message = system_message
        self.base_url = base_url
        self.api_key = api_key
        self._rate_limiter = _RateLimiter(rpm)
        self.total_cost = 0.0
        self.total_tokens = 0

    def _build_messages(self, prompt: str, system_message: str | None) -> list[dict]:
        messages = []
        sys_msg = system_message or self.system_message
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _track_usage(self, response) -> None:
        try:
            usage = response.usage
            if usage:
                self.total_tokens += getattr(usage, "total_tokens", 0)
            cost = litellm.completion_cost(completion_response=response)
            if cost:
                self.total_cost += cost
        except Exception:
            pass

    def _get_completion_kwargs(self, model: str) -> dict:
        kwargs: dict = {"model": model, "timeout": self.timeout}
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    def _model_chain(self) -> list[str]:
        return [self.model] + self.fallback_models

    def call(self, prompt: str, system_message: str | None = None) -> str:
        """Call the LLM with automatic fallback and retry. Returns text response."""
        messages = self._build_messages(prompt, system_message)

        for model in self._model_chain():
            for attempt in range(self.max_retries):
                self._rate_limiter.wait_sync()
                try:
                    kwargs = self._get_completion_kwargs(model)
                    response = litellm.completion(messages=messages, **kwargs)
                    self._track_usage(response)
                    return response.choices[0].message.content.strip()
                except litellm.RateLimitError:
                    wait = self.rate_limit_base_wait * (2 ** attempt)
                    logger.warning(f"Rate limited on {model}, waiting {wait:.0f}s")
                    time.sleep(wait)
                except Exception as e:
                    logger.warning(f"LLM call failed ({model}, attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        logger.info(f"Exhausted retries for {model}, trying next fallback")
                        break

        raise RuntimeError("All models in fallback chain failed")

    def call_json(self, prompt: str, system_message: str | None = None) -> dict:
        """Call LLM and parse response as JSON."""
        text = self.call(prompt, system_message)
        return parse_llm_json(text)

    async def acall(self, prompt: str, system_message: str | None = None) -> str:
        """Async version of call() with rate limiting and fallback."""
        messages = self._build_messages(prompt, system_message)

        for model in self._model_chain():
            for attempt in range(self.max_retries):
                await self._rate_limiter.wait_async()
                try:
                    kwargs = self._get_completion_kwargs(model)
                    response = await litellm.acompletion(messages=messages, **kwargs)
                    self._track_usage(response)
                    return response.choices[0].message.content.strip()
                except litellm.RateLimitError:
                    wait = self.rate_limit_base_wait * (2 ** attempt)
                    logger.warning(f"Rate limited on {model}, waiting {wait:.0f}s")
                    await asyncio.sleep(wait)
                except Exception as e:
                    logger.warning(f"Async LLM call failed ({model}, attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        logger.info(f"Exhausted retries for {model}, trying next fallback")
                        break

        raise RuntimeError("All models in fallback chain failed")

    async def acall_json(self, prompt: str, system_message: str | None = None) -> dict:
        """Async version of call_json()."""
        text = await self.acall(prompt, system_message)
        return parse_llm_json(text)


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, handling common quirks.

    LLMs often wrap JSON in markdown code fences or include
    trailing explanation text. This function strips that.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ``
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try parsing as-is
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object in the text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")
