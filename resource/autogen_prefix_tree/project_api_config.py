from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass(frozen=True)
class ProjectProviderConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    config_path: str | None

    def redacted_marker(self) -> str:
        source = "env" if self.config_path is None else "config/config.txt"
        return f"project DeepSeek config ({source}, model={self.model}, base_url={self.base_url})"


def load_project_provider_config(
    *,
    cwd: str | Path = ".",
    env: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> ProjectProviderConfig | None:
    effective_env = os.environ if env is None else env
    root = Path(cwd)
    candidates: list[Path] = []
    if config_path is not None:
        explicit_path = Path(config_path)
        candidates.append(explicit_path if explicit_path.is_absolute() else root / explicit_path)
    candidates.append(root / "config" / "config.txt")

    parsed: dict[str, str] = {}
    selected_path: Path | None = None

    for path in candidates:
        if not path.exists():
            continue
        selected_path = path.resolve()
        parsed = _parse_config_file(path)
        break

    api_key = _first_nonempty(parsed.get("api_key"), effective_env.get("DEEPSEEK_API_KEY"))
    model = _first_nonempty(parsed.get("model"), effective_env.get("DEEPSEEK_MODEL"))
    base_url = _first_nonempty(
        parsed.get("base_url"),
        effective_env.get("DEEPSEEK_BASE_URL"),
        effective_env.get("OPENAI_BASE_URL"),
    )

    if not api_key:
        return None
    return ProjectProviderConfig(
        provider="deepseek",
        api_key=api_key,
        model=normalize_deepseek_model(model or ""),
        base_url=(base_url or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/"),
        config_path=str(selected_path) if selected_path is not None else None,
    )


def project_runtime_env(
    *,
    cwd: str | Path = ".",
    env: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    runtime_env = dict(os.environ if env is None else env)
    config = load_project_provider_config(cwd=cwd, env=runtime_env, config_path=config_path)
    if config is None:
        return runtime_env
    runtime_env.setdefault("DEEPSEEK_API_KEY", config.api_key)
    runtime_env.setdefault("DEEPSEEK_MODEL", config.model)
    runtime_env.setdefault("DEEPSEEK_BASE_URL", config.base_url)
    runtime_env.setdefault("OPENAI_API_KEY", config.api_key)
    return runtime_env


def normalize_deepseek_model(model: str) -> str:
    value = (model or "").strip().lower().replace("_", "-")
    aliases = {
        "": "deepseek-v4-pro",
        "deepseek": "deepseek-v4-pro",
        "ds": "deepseek-v4-pro",
        "deepseek-v4pro": "deepseek-v4-pro",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4flash": "deepseek-v4-flash",
        "deepseek-v4-flash": "deepseek-v4-flash",
    }
    return aliases.get(value, model.strip() or "deepseek-v4-pro")


def _parse_config_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = ""
        value = stripped
        if ":" in stripped:
            name, value = stripped.split(":", 1)
        elif "=" in stripped:
            name, value = stripped.split("=", 1)
        normalized_name = name.strip().lower()
        value = value.strip()
        if not value:
            continue
        if normalized_name in {"deepseek", "deepseek_api_key", "api_key"}:
            parsed.setdefault("api_key", value)
        elif normalized_name in {"model", "deepseek_model"}:
            parsed.setdefault("model", value)
        elif normalized_name in {"base_url", "deepseek_base_url", "openai_base_url"}:
            parsed.setdefault("base_url", value)
        elif not normalized_name:
            parsed.setdefault("model", value)
    return parsed


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""
