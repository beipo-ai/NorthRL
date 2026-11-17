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

"""Small standalone Serper search command for the JobBench agent's Bash tool."""

import argparse
import json
import os
import sys
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_DEFAULT_API_URL = "https://google.serper.dev/search"
_TIMEOUT_SECONDS = 30
_MAX_RESPONSE_BYTES = 1024 * 1024


class SearchError(Exception):
    """An actionable search failure safe to show in an agent's tool output."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward X-API-KEY to a redirect target, including same-host URLs.
        return None


def _validate_api_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() or ord(character) < 32 for character in value)
        )
        # Accessing port also rejects malformed ports before credentials are used.
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise SearchError("SERPER_API_URL must be an HTTPS URL without credentials or query data")
    return value


def search(query: str, num: int = 5) -> dict:
    """Return a bounded list of titles, source URLs and snippets from Serper."""
    if not isinstance(query, str) or not query.strip():
        raise SearchError("Search query must be nonempty text")
    if isinstance(num, bool) or not isinstance(num, int) or not 1 <= num <= 10:
        raise SearchError("Result count must be an integer from 1 to 10")
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        raise SearchError("SERPER_API_KEY is not configured")
    if any(ord(character) < 32 or ord(character) > 126 for character in api_key):
        raise SearchError("SERPER_API_KEY has an invalid format")
    api_url = _validate_api_url(os.environ.get("SERPER_API_URL", _DEFAULT_API_URL).strip())
    request = Request(
        api_url,
        data=json.dumps({"q": query, "num": num}).encode("utf-8"),
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with build_opener(_NoRedirects()).open(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        status = exc.code
        exc.close()
        if status in (401, 403):
            message = "Serper rejected authentication; check the configured API key"
        elif status == 429:
            message = "Serper quota or rate limit exceeded; check account credits and retry later"
        elif 300 <= status < 400:
            message = "Serper redirect refused; configure the final HTTPS search endpoint"
        else:
            message = f"Serper request failed with HTTP status {status}"
        raise SearchError(message) from None
    except (URLError, OSError, HTTPException, ValueError):
        raise SearchError(
            "Serper request failed or timed out; check network and endpoint"
        ) from None
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise SearchError("Serper response exceeded the size limit")
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        raise SearchError("Serper returned invalid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("organic"), list):
        raise SearchError("Serper returned an invalid search response")
    results = []
    for item in data["organic"][:num]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(name), str) or not item[name].strip()
            for name in ("title", "link")
        ):
            raise SearchError("Serper returned an invalid search result")
        snippet = item.get("snippet", "")
        if not isinstance(snippet, str):
            raise SearchError("Serper returned an invalid search result")
        results.append({"title": item["title"], "url": item["link"], "snippet": snippet})
    return {"query": query, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the web using Serper.dev")
    parser.add_argument("query", help="Search query, quoted as one argument")
    parser.add_argument("--num", type=int, choices=range(1, 11), default=5)
    args = parser.parse_args(argv)
    try:
        result = search(args.query, args.num)
    except SearchError as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
