# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded asynchronous backoff without changing a verifier's failure policy."""

import logging
import math
import random
import ssl
import time
from asyncio import sleep
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

import openai

MAX_JUDGE_ATTEMPTS = 5  # Initial request plus at most four retries.
FINAL_JUDGE_TIMEOUT_S = 300.0
MAX_SERVER_RETRY_AFTER_S = 60.0
_PERMANENT_QUOTA_CODES = {"insufficient_quota", "billing_hard_limit_reached"}
_CONTEXT_FIELDS = ("reward_fn", "task_id", "group_id", "sample_id", "judge_model", "judge_mode")


def judge_timeout_for_attempt(base_timeout_s: float, attempt: int) -> float:
    """Select a request-local timeout using a one-based attempt number."""
    if not 1 <= attempt <= MAX_JUDGE_ATTEMPTS:
        raise ValueError("attempt must be within the judge attempt budget")
    return FINAL_JUDGE_TIMEOUT_S if attempt == MAX_JUDGE_ATTEMPTS else base_timeout_s


def _is_retryable(error: Exception) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError) or any(
            marker in str(current).lower()
            for marker in ("certificate_verify_failed", "certificate verify failed")
        ):
            return False
        current = current.__cause__ or current.__context__

    if isinstance(error, openai.APIStatusError):
        body = error.body
        if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
            body = body["error"]
        codes = [getattr(error, "code", None), getattr(error, "type", None)]
        if isinstance(body, Mapping):
            codes.extend((body.get("code"), body.get("type")))
        if any(isinstance(code, str) and code in _PERMANENT_QUOTA_CODES for code in codes):
            return False
        return error.status_code in (408, 409, 429) or error.status_code >= 500

    # Keep the callers' existing retries for malformed API responses / parser
    # exceptions. Each caller still decides whether a missing verdict is an
    # exception (Math) or a scored result (Recast / Reasoning360).
    return True


def _retry_after_seconds(error: Exception) -> float | None:
    if not isinstance(error, openai.APIStatusError):
        return None
    value = error.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = max(0.0, date.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


async def wait_for_judge_retry(
    error: Exception,
    *,
    failed_attempt: int,
    judge_name: str,
    logger: logging.Logger,
    elapsed_s: float,
    timeout_s: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> bool:
    """Wait before another attempt, or return False for caller-owned termination.

    Attempts are one-based. Without a server hint waits are 1-3s, 2-6s, 4-12s, 8-24s.
    SDK retries must remain disabled. Cancellation during sleep propagates.
    """
    if not 1 <= failed_attempt <= MAX_JUDGE_ATTEMPTS:
        raise ValueError("failed_attempt must be within the judge attempt budget")
    safe_context = {key: context[key] for key in _CONTEXT_FIELDS if context and key in context}
    retry_after = _retry_after_seconds(error)
    stop_reason = None
    if failed_attempt == MAX_JUDGE_ATTEMPTS:
        stop_reason = "attempts_exhausted"
    elif not _is_retryable(error):
        stop_reason = "non_retryable_error"
    elif retry_after is not None and retry_after > MAX_SERVER_RETRY_AFTER_S:
        stop_reason = "retry_after_exceeds_budget"
    if stop_reason is not None:
        logger.warning(
            "%s judge request failed; not retrying: attempt=%d/%d error_type=%s "
            "http_status=%r elapsed_s=%.3f timeout_s=%r stop_reason=%s "
            "retry_after_s=%r context=%r",
            judge_name,
            failed_attempt,
            MAX_JUDGE_ATTEMPTS,
            type(error).__name__,
            getattr(error, "status_code", None),
            elapsed_s,
            timeout_s,
            stop_reason,
            retry_after,
            safe_context,
        )
        return False

    delay = random.uniform(1.0, 3.0) * 2 ** (failed_attempt - 1)
    if retry_after is not None:
        # Respect the server's minimum, retaining jitter to spread out retries.
        delay += retry_after
    logger.warning(
        "%s judge request failed; retrying: attempt=%d/%d error_type=%s "
        "http_status=%r elapsed_s=%.3f timeout_s=%r delay_s=%.3f "
        "retry_after_s=%r context=%r",
        judge_name,
        failed_attempt,
        MAX_JUDGE_ATTEMPTS,
        type(error).__name__,
        getattr(error, "status_code", None),
        elapsed_s,
        timeout_s,
        delay,
        retry_after,
        safe_context,
    )
    await sleep(delay)
    return True
