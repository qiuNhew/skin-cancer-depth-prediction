"""YAML configuration loading, saving, and access utilities.

Each experiment is defined by one YAML file (V1 -> config_v1.yaml,
V2 -> config_v2.yaml). The plain-dict API (`load_config` / `save_config` /
`get_config_value` / `merge_configs` / `resolve_paths`) is what train.py and
evaluate.py use. The `Config` class is an attribute-style wrapper for callers
that prefer `cfg.training.batch_size` over `cfg["training"]["batch_size"]`.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML file into a plain dict. UTF-8 (handles µm in headers).

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed nested dict.

    Raises:
        FileNotFoundError: file missing.
        yaml.YAMLError:    file present but parse error.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # `yaml.safe_load` (not `load`) — refuses to instantiate arbitrary Python objects.
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Serialise a config dict to YAML. Used by train.py to snapshot results/config_used.yaml."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # block style + preserve key order -> diff-friendly output
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Look up a nested config value with dot notation. Returns `default` if any segment missing.

    Example:
        >>> get_config_value({"training": {"batch_size": 4}}, "training.batch_size")
        4
        >>> get_config_value({"training": {}}, "training.lr", default=1e-4)
        0.0001
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def resolve_paths(
    config: Dict[str, Any],
    project_root: Union[str, Path],
) -> Dict[str, Any]:
    """Return a deep copy of `config` with relative paths resolved against `project_root`.

    Resolves `data.data_dir` and `paths.{checkpoint_dir, results_dir, log_dir}`.
    Absolute values pass through unchanged; missing keys are left missing.
    Lets entry-point scripts run from any cwd: relative paths in the YAML are
    interpreted relative to the project root, not the shell's working directory.

    Always pass the *unresolved* config to `save_config` for portable snapshots.
    """
    project_root = Path(project_root).resolve()
    resolved = copy.deepcopy(config)

    def _to_absolute(value: Any) -> str:
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((project_root / p).resolve())

    data = resolved.get("data")
    if isinstance(data, dict) and data.get("data_dir") is not None:
        data["data_dir"] = _to_absolute(data["data_dir"])

    paths = resolved.get("paths")
    if isinstance(paths, dict):
        for key in ("checkpoint_dir", "results_dir", "log_dir"):
            if paths.get(key) is not None:
                paths[key] = _to_absolute(paths[key])

    return resolved


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts; `override` values win on conflict. Returns NEW dict.

    Useful for an "inheritance" pattern (base config + per-experiment deltas).
    Not used by V1/V2 currently — each config is self-contained.

    Example:
        >>> merge_configs({"training": {"batch_size": 4, "lr": 1e-4}}, {"training": {"lr": 5e-4}})
        {'training': {'batch_size': 4, 'lr': 0.0005}}
    """
    result = base_config.copy()
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Attribute-style wrapper around a config dict (`cfg.training.batch_size`).

    Recursively wraps nested dicts so chained attribute access works at any depth.
    Most of the project uses the plain-dict `load_config` API; this class is for notebook use.

    Args:
        config_path: Optional YAML path to load.
        config_dict: Optional dict to initialise from (wins if both given).

    Example:
        >>> cfg = Config("configs/config_v1.yaml")
        >>> cfg.training.batch_size
        4
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        if config_path is not None:
            self._config = load_config(config_path)
        elif config_dict is not None:
            self._config = config_dict
        else:
            self._config = {}

        # Recursively wrap nested dicts so chained attribute access works.
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(config_dict=value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like `.get()` with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Unwrap back to a plain dict (drops attribute access)."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config({self._config})"


# Default config = V1 baseline.
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config_v1.yaml"


def get_default_config() -> Config:
    """Load the V1 baseline config as a `Config` object. Convenience for notebooks/smoke tests."""
    if DEFAULT_CONFIG_PATH.exists():
        return Config(DEFAULT_CONFIG_PATH)
    raise FileNotFoundError(f"Default config not found at {DEFAULT_CONFIG_PATH}")
