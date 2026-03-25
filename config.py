"""Configuration management for the Forex Trading Agent."""

import json
import os
from pathlib import Path
from typing import Any, Dict


_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Central configuration object.

    Loads ``config/default.json`` and optionally merges a user-supplied
    override file whose path is given by the ``FOREX_CONFIG`` environment
    variable.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = _load_json(_DEFAULT_CONFIG_PATH)

        env_path = os.environ.get("FOREX_CONFIG")
        if env_path:
            override_path = Path(env_path)
            if override_path.exists():
                self._data = _deep_merge(self._data, _load_json(override_path))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def trading(self) -> Dict[str, Any]:
        return self._data["trading"]

    @property
    def indicators(self) -> Dict[str, Any]:
        return self._data["indicators"]

    @property
    def signals(self) -> Dict[str, Any]:
        return self._data["signals"]

    @property
    def data(self) -> Dict[str, Any]:
        return self._data["data"]

    @property
    def api(self) -> Dict[str, Any]:
        return self._data["api"]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# Singleton instance
config = Config()
