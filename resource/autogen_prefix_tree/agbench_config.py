from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_capture_model_config(
    base_model_config: Mapping[str, Any],
    *,
    capture_log_path: str,
    session_id: str,
    include_message_content: bool = True,
) -> dict[str, Any]:
    return {
        "provider": "autogen_prefix_tree.request_capture.RequestCaptureClient",
        "component_type": "model",
        "config": {
            "inner_client": deepcopy(dict(base_model_config)),
            "capture_log_path": capture_log_path,
            "session_id": session_id,
            "include_message_content": include_message_content,
        },
    }


def build_prefix_reorder_model_config(
    base_model_config: Mapping[str, Any],
    *,
    telemetry_log_path: str,
    session_id: str,
    capture_log_path: str | None = None,
    include_message_content: bool = True,
    enabled: bool = True,
    enable_natural_language_segmentation: bool = False,
    min_estimated_gain_chars: int = 1,
    semantic_guard_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inner_client = deepcopy(dict(base_model_config))
    if capture_log_path is not None:
        inner_client = build_capture_model_config(
            inner_client,
            capture_log_path=capture_log_path,
            session_id=session_id,
            include_message_content=include_message_content,
        )
    config: dict[str, Any] = {
        "inner_client": inner_client,
        "session_id": session_id,
        "enabled": enabled,
        "enable_natural_language_segmentation": enable_natural_language_segmentation,
        "telemetry_log_path": telemetry_log_path,
        "min_estimated_gain_chars": min_estimated_gain_chars,
    }
    if semantic_guard_config is not None:
        config["semantic_guard"] = deepcopy(dict(semantic_guard_config))
    return {
        "provider": "autogen_prefix_tree.client.PrefixReorderClient",
        "component_type": "model",
        "config": config,
    }


def build_static_capture_model_config(
    *,
    capture_log_path: str,
    session_id: str,
    responses: Sequence[str] = ("TERMINATE",),
    include_message_content: bool = True,
) -> dict[str, Any]:
    return build_capture_model_config(
        {
            "provider": "autogen_prefix_tree.static_client.StaticResponseClient",
            "component_type": "model",
            "config": {
                "responses": list(responses),
                "model": "static-agbench-smoke",
            },
        },
        capture_log_path=capture_log_path,
        session_id=session_id,
        include_message_content=include_message_content,
    )


def build_agbench_config(
    base_config: Mapping[str, Any],
    *,
    mode: str,
    session_id: str,
    capture_log_path: str | None = None,
    telemetry_log_path: str | None = None,
    include_message_content: bool = True,
    enabled: bool = True,
    enable_natural_language_segmentation: bool = False,
    semantic_guard_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = deepcopy(dict(base_config))
    base_model_config = _extract_model_config(result)
    if mode == "baseline":
        result["model_config"] = base_model_config
    elif mode == "capture":
        if capture_log_path is None:
            raise ValueError("capture_log_path is required for capture mode")
        result["model_config"] = build_capture_model_config(
            base_model_config,
            capture_log_path=capture_log_path,
            session_id=session_id,
            include_message_content=include_message_content,
        )
    elif mode == "plugin":
        if telemetry_log_path is None:
            raise ValueError("telemetry_log_path is required for plugin mode")
        result["model_config"] = build_prefix_reorder_model_config(
            base_model_config,
            telemetry_log_path=telemetry_log_path,
            session_id=session_id,
            capture_log_path=capture_log_path,
            include_message_content=include_message_content,
            enabled=enabled,
            enable_natural_language_segmentation=enable_natural_language_segmentation,
            semantic_guard_config=semantic_guard_config,
        )
    elif mode == "static_capture":
        if capture_log_path is None:
            raise ValueError("capture_log_path is required for static_capture mode")
        result["model_config"] = build_static_capture_model_config(
            capture_log_path=capture_log_path,
            session_id=session_id,
            include_message_content=include_message_content,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AutoGenBench model_config wrappers for prefix evaluation.")
    parser.add_argument("--base-config", required=True, help="Input YAML/JSON config containing model_config.")
    parser.add_argument("--output", required=True, help="Output YAML/JSON config path.")
    parser.add_argument("--mode", choices=("baseline", "capture", "plugin", "static_capture"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--capture-log-path")
    parser.add_argument("--telemetry-log-path")
    parser.add_argument("--redact-message-content", action="store_true")
    parser.add_argument("--disabled", action="store_true", help="Generate disabled PrefixReorderClient config.")
    parser.add_argument(
        "--enable-natural-language-segmentation",
        action="store_true",
        help="Generate plugin config with experimental natural-language system prompt segmentation enabled.",
    )
    parser.add_argument("--semantic-guard-base-url", help="Optional local OpenAI-compatible semantic guard base URL.")
    parser.add_argument("--semantic-guard-model", help="Model name for --semantic-guard-base-url.")
    parser.add_argument("--semantic-guard-timeout", type=float, default=30.0)
    parser.add_argument("--semantic-guard-min-confidence", type=float, default=0.75)
    parser.add_argument("--semantic-guard-max-message-chars", type=int, default=12000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = _read_structured(Path(args.base_config))
    generated = build_agbench_config(
        base_config,
        mode=args.mode,
        session_id=args.session_id,
        capture_log_path=args.capture_log_path,
        telemetry_log_path=args.telemetry_log_path,
        include_message_content=not args.redact_message_content,
        enabled=not args.disabled,
        enable_natural_language_segmentation=args.enable_natural_language_segmentation,
        semantic_guard_config=_semantic_guard_config_from_args(args),
    )
    _write_structured(Path(args.output), generated)
    return 0


def _semantic_guard_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.semantic_guard_base_url:
        return None
    if not args.semantic_guard_model:
        raise ValueError("--semantic-guard-model is required when --semantic-guard-base-url is set")
    return {
        "provider": "openai-compatible",
        "base_url": args.semantic_guard_base_url,
        "model": args.semantic_guard_model,
        "timeout_seconds": args.semantic_guard_timeout,
        "min_confidence": args.semantic_guard_min_confidence,
        "max_message_chars": args.semantic_guard_max_message_chars,
    }


def _extract_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("model_config")
    if not isinstance(value, Mapping):
        raise ValueError("base config must contain a model_config mapping")
    return deepcopy(dict(value))


def _read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("config file must contain a mapping")
    return value


def _write_structured(path: Path, value: Mapping[str, Any]) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write YAML configs") from exc
    path.write_text(yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
