"""
Shared retry policy for direct OpenAI SDK calls.

Without this, a single transient RateLimitError or APITimeoutError from
OpenAI anywhere in the 9-node pipeline fails the user's entire request,
even though the next attempt (a second later) would likely succeed.

Usage: decorate the function that makes ONE client.*.create(...) call.
    @with_llm_retry
    def _call():
        return _client.chat.completions.create(...)
    resp = _call()
"""
from __future__ import annotations
import logging

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential, before_sleep_log)

logger = logging.getLogger(__name__)

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

with_llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type(RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
