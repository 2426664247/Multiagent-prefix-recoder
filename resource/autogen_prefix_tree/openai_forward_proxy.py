from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from .openai_request_adapter import OpenAICompatibleRequestAdapter
from .project_api_config import load_project_provider_config
from .semantic_guard import OpenAICompatibleSemanticGuard
from .telemetry import JsonlTelemetryLogger
from .validator import CacheUtilityValidator


@dataclass(frozen=True)
class ForwardProxyConfig:
    upstream_base_url: str
    session_id: str = "openai-forward-proxy"
    telemetry_log_path: str | None = None
    enabled: bool = True
    enable_natural_language_segmentation: bool = False
    enable_groupchat_history_reordering: bool = False
    timeout_seconds: float = 120.0
    semantic_guard_base_url: str | None = None
    semantic_guard_model: str | None = None
    semantic_guard_timeout: float = 30.0
    semantic_guard_min_confidence: float = 0.75
    semantic_guard_max_message_chars: int = 12000
    shadow_trial_plan_path: str | None = None
    upstream_api_key: str | None = None
    reset_telemetry_log: bool = False


def build_forward_proxy_server(
    *,
    host: str,
    port: int,
    upstream_base_url: str,
    session_id: str = "openai-forward-proxy",
    telemetry_log_path: str | Path | None = None,
    enabled: bool = True,
    enable_natural_language_segmentation: bool = False,
    enable_groupchat_history_reordering: bool = False,
    timeout_seconds: float = 120.0,
    semantic_guard_base_url: str | None = None,
    semantic_guard_model: str | None = None,
    semantic_guard_timeout: float = 30.0,
    semantic_guard_min_confidence: float = 0.75,
    semantic_guard_max_message_chars: int = 12000,
    shadow_trial_plan_path: str | Path | None = None,
    upstream_api_key: str | None = None,
    upstream_api_key_env: str | None = None,
    use_project_deepseek_config: bool = False,
    project_config_path: str | Path | None = None,
    reset_telemetry_log: bool = False,
) -> ThreadingHTTPServer:
    resolved_upstream_api_key = _resolve_upstream_api_key(
        upstream_api_key=upstream_api_key,
        upstream_api_key_env=upstream_api_key_env,
        use_project_deepseek_config=use_project_deepseek_config,
        project_config_path=project_config_path,
    )
    if reset_telemetry_log and telemetry_log_path is not None:
        _reset_telemetry_log_path(telemetry_log_path)
    config = ForwardProxyConfig(
        upstream_base_url=upstream_base_url,
        session_id=session_id,
        telemetry_log_path=str(telemetry_log_path) if telemetry_log_path is not None else None,
        enabled=enabled,
        enable_natural_language_segmentation=enable_natural_language_segmentation,
        enable_groupchat_history_reordering=enable_groupchat_history_reordering,
        timeout_seconds=timeout_seconds,
        semantic_guard_base_url=semantic_guard_base_url,
        semantic_guard_model=semantic_guard_model,
        semantic_guard_timeout=semantic_guard_timeout,
        semantic_guard_min_confidence=semantic_guard_min_confidence,
        semantic_guard_max_message_chars=semantic_guard_max_message_chars,
        shadow_trial_plan_path=str(shadow_trial_plan_path) if shadow_trial_plan_path is not None else None,
        upstream_api_key=resolved_upstream_api_key,
        reset_telemetry_log=reset_telemetry_log,
    )
    adapter = OpenAICompatibleRequestAdapter(
        session_id=session_id,
        telemetry_log_path=telemetry_log_path,
        validator=_validator_from_config(config),
        enabled=enabled,
        enable_natural_language_segmentation=enable_natural_language_segmentation,
        enable_groupchat_history_reordering=enable_groupchat_history_reordering,
        shadow_trial_plan_path=config.shadow_trial_plan_path,
    )

    class Handler(_ForwardProxyHandler):
        proxy_config = config
        request_adapter = adapter

    return ThreadingHTTPServer((host, port), Handler)


class _ForwardProxyHandler(BaseHTTPRequestHandler):
    proxy_config: ForwardProxyConfig
    request_adapter: OpenAICompatibleRequestAdapter

    def do_GET(self) -> None:
        self._write_json(404, {"error": {"message": "Only POST /v1/chat/completions is supported."}})

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self._write_json(404, {"error": {"message": "Only /v1/chat/completions is supported."}})
            return
        try:
            body = self._read_json_body()
            if not isinstance(body, Mapping):
                self._write_json(400, {"error": {"message": "Request body must be a JSON object."}})
                return
            rewrite_result = self.request_adapter.rewrite_request_body(
                body,
                operation="forward_proxy.chat.completions",
            )
            started = time.perf_counter()
            status, headers, response_body = self._forward_json(rewrite_result.rewritten_body)
            latency_seconds = time.perf_counter() - started
            self._write_provider_telemetry(
                request_index=int(rewrite_result.telemetry.get("request_index") or 0),
                rewrite_applied=rewrite_result.applied,
                rewrite_reason=str((rewrite_result.telemetry.get("validation") or {}).get("reason") or "unknown"),
                status_code=status,
                latency_seconds=latency_seconds,
                response_body=response_body,
                error=None,
            )
            self._write_raw(status, headers, response_body)
        except json.JSONDecodeError:
            self._write_json(400, {"error": {"message": "Invalid JSON request body."}})
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            self._write_provider_telemetry(
                request_index=0,
                rewrite_applied=False,
                rewrite_reason="upstream_http_error",
                status_code=exc.code,
                latency_seconds=0.0,
                response_body=response_body,
                error=f"HTTPError:{exc.code}",
            )
            self._write_raw(exc.code, _response_headers(exc.headers), response_body)
        except urllib.error.URLError as exc:
            self._write_provider_telemetry(
                request_index=0,
                rewrite_applied=False,
                rewrite_reason="upstream_url_error",
                status_code=502,
                latency_seconds=0.0,
                response_body=b"",
                error=f"URLError:{exc.reason}",
            )
            self._write_json(502, {"error": {"message": f"Upstream request failed: {exc.reason}"}})
        except Exception as exc:  # noqa: BLE001
            self._write_json(500, {"error": {"message": f"Proxy error: {type(exc).__name__}"}})

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _forward_json(self, body: Mapping[str, Any]) -> tuple[int, Mapping[str, str], bytes]:
        request = urllib.request.Request(
            _join_url(self.proxy_config.upstream_base_url, self.path),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._forward_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.proxy_config.timeout_seconds) as response:
            return response.status, _response_headers(response.headers), response.read()

    def _forward_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        authorization = self.headers.get("Authorization")
        if self.proxy_config.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.proxy_config.upstream_api_key}"
        elif authorization:
            headers["Authorization"] = authorization
        return headers

    def _write_json(self, status: int, value: Mapping[str, Any]) -> None:
        self._write_raw(
            status,
            {"Content-Type": "application/json"},
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
        )

    def _write_raw(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            lowered = key.lower()
            if lowered in {"connection", "transfer-encoding", "content-encoding", "content-length"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_provider_telemetry(
        self,
        *,
        request_index: int,
        rewrite_applied: bool,
        rewrite_reason: str,
        status_code: int,
        latency_seconds: float,
        response_body: bytes,
        error: str | None,
    ) -> None:
        if not self.proxy_config.telemetry_log_path:
            return
        usage = _extract_usage(response_body)
        record: dict[str, Any] = {
            "schema_version": "prefix-forward-proxy-provider-telemetry-v1",
            "session_id": self.proxy_config.session_id,
            "request_index": request_index,
            "operation": "forward_proxy.provider_response",
            "rewrite_applied": rewrite_applied,
            "rewrite_reason": rewrite_reason,
            "upstream_status": status_code,
            "latency_seconds": latency_seconds,
            "error": error,
            "actual_prompt_tokens": usage["prompt_tokens"],
            "actual_cached_tokens": usage["cached_tokens"],
            "actual_completion_tokens": usage["completion_tokens"],
            "actual_total_tokens": usage["total_tokens"],
        }
        if usage["cost_usd"] is not None:
            record["actual_cost_usd"] = usage["cost_usd"]
        JsonlTelemetryLogger(self.proxy_config.telemetry_log_path)(record)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible prefix-reorder forwarding proxy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--session-id", default="openai-forward-proxy")
    parser.add_argument("--telemetry")
    parser.add_argument("--disabled", action="store_true", help="Forward without rewriting, but still record telemetry.")
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Experimental opt-in for unmarked natural-language system prompts.",
    )
    parser.add_argument(
        "--enable-groupchat-history-reordering",
        action="store_true",
        help="Experimental opt-in for exact repeated groupchat dialogue prefix reordering.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--semantic-guard-base-url", help="Optional local OpenAI-compatible semantic guard base URL.")
    parser.add_argument("--semantic-guard-model", help="Model name for --semantic-guard-base-url.")
    parser.add_argument("--semantic-guard-timeout", type=float, default=30.0)
    parser.add_argument("--semantic-guard-min-confidence", type=float, default=0.75)
    parser.add_argument("--semantic-guard-max-message-chars", type=int, default=12000)
    parser.add_argument(
        "--shadow-trial-plan",
        help=(
            "Optional prompt-safe static_rule_calibration_eval summary or shadow_trial_plan JSON. "
            "Records shadow-only Validator rule matches in adapter telemetry without changing forwarded requests."
        ),
    )
    parser.add_argument(
        "--upstream-api-key-env",
        help="Optional environment variable whose value should be used as the upstream Bearer token.",
    )
    parser.add_argument(
        "--use-project-deepseek-config",
        action="store_true",
        help="Read config/config.txt or DEEPSEEK_API_KEY and use that DeepSeek key for upstream Authorization.",
    )
    parser.add_argument("--project-config", help="Optional project provider config path for --use-project-deepseek-config.")
    parser.add_argument(
        "--reset-telemetry",
        action="store_true",
        help="Truncate the telemetry JSONL at startup so a real A/B run does not mix with prior smoke rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = build_forward_proxy_server(
        host=args.host,
        port=args.port,
        upstream_base_url=args.upstream_base_url,
        session_id=args.session_id,
        telemetry_log_path=args.telemetry,
        enabled=not args.disabled,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
        enable_groupchat_history_reordering=args.enable_groupchat_history_reordering,
        timeout_seconds=args.timeout_seconds,
        semantic_guard_base_url=args.semantic_guard_base_url,
        semantic_guard_model=args.semantic_guard_model,
        semantic_guard_timeout=args.semantic_guard_timeout,
        semantic_guard_min_confidence=args.semantic_guard_min_confidence,
        semantic_guard_max_message_chars=args.semantic_guard_max_message_chars,
        shadow_trial_plan_path=args.shadow_trial_plan,
        upstream_api_key_env=args.upstream_api_key_env,
        use_project_deepseek_config=args.use_project_deepseek_config,
        project_config_path=args.project_config,
        reset_telemetry_log=args.reset_telemetry,
    )
    print(f"Serving OpenAI-compatible prefix proxy on http://{args.host}:{server.server_port}")
    print(f"Forwarding to {args.upstream_base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        threading.Thread(target=server.shutdown, daemon=True).start()
        server.server_close()
    return 0


def _validator_from_config(config: ForwardProxyConfig) -> CacheUtilityValidator:
    if not config.semantic_guard_base_url:
        return CacheUtilityValidator()
    if not config.semantic_guard_model:
        raise ValueError("semantic_guard_model is required when semantic_guard_base_url is set")
    return CacheUtilityValidator(
        semantic_guard=OpenAICompatibleSemanticGuard(
            base_url=config.semantic_guard_base_url,
            model=config.semantic_guard_model,
            timeout_seconds=config.semantic_guard_timeout,
            min_confidence=config.semantic_guard_min_confidence,
            max_message_chars=config.semantic_guard_max_message_chars,
        )
    )


def _resolve_upstream_api_key(
    *,
    upstream_api_key: str | None,
    upstream_api_key_env: str | None,
    use_project_deepseek_config: bool,
    project_config_path: str | Path | None,
) -> str | None:
    if upstream_api_key:
        return upstream_api_key
    if upstream_api_key_env:
        value = os.environ.get(upstream_api_key_env)
        if value and value.strip():
            return value.strip()
    if use_project_deepseek_config:
        config = load_project_provider_config(cwd=Path.cwd(), config_path=project_config_path)
        if config is None:
            raise ValueError("project DeepSeek config not found; set DEEPSEEK_API_KEY or create config/config.txt")
        return config.api_key
    return None


def _reset_telemetry_log_path(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    route = "/" + path.lstrip("/")
    if base.endswith("/v1") and route.startswith("/v1/"):
        route = route[len("/v1") :]
    return base + route


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in headers.items()}


def _extract_usage(response_body: bytes) -> dict[str, int | float | None]:
    usage: dict[str, int | float | None] = {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
    }
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return usage
    if not isinstance(payload, Mapping):
        return usage
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return usage
    usage["prompt_tokens"] = _int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"))
    usage["completion_tokens"] = _int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"))
    usage["total_tokens"] = _int(raw_usage.get("total_tokens"))
    if not usage["total_tokens"]:
        usage["total_tokens"] = int(usage["prompt_tokens"] or 0) + int(usage["completion_tokens"] or 0)
    usage["cached_tokens"] = _cached_tokens(raw_usage)
    usage["cost_usd"] = _cost_usd(raw_usage)
    return usage


def _cached_tokens(raw_usage: Mapping[str, Any]) -> int:
    for key in (
        "cached_tokens",
        "cached_prompt_tokens",
        "prompt_cache_hit_tokens",
        "cache_hit_tokens",
        "input_cached_tokens",
    ):
        value = _int(raw_usage.get(key))
        if value:
            return value
    for detail_key in ("prompt_tokens_details", "input_token_details", "input_tokens_details"):
        details = raw_usage.get(detail_key)
        if isinstance(details, Mapping):
            value = _cached_tokens_from_details(details)
            if value:
                return value
    return 0


def _cached_tokens_from_details(details: Mapping[str, Any]) -> int:
    for key in (
        "cached_tokens",
        "cache_read",
        "cached",
        "cache_hit_tokens",
        "prompt_cache_hit_tokens",
    ):
        value = _int(details.get(key))
        if value:
            return value
    return 0


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _cost_usd(raw_usage: Mapping[str, Any]) -> float | None:
    for key in ("cost_usd", "cost", "total_cost_usd", "total_cost"):
        value = _float(raw_usage.get(key))
        if value or key in raw_usage:
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
