"""Explicit configuration. No hidden global state, no secrets in source.

Resolution order for every setting: CLI flag > environment variable >
config file (agentwb.json) > built-in default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONFIG_NAMES = ("agentwb.json", ".agentwb.json")


@dataclass
class Settings:
    data_dir: Path = Path("data")
    provider: str = "mock"
    model: Optional[str] = None
    tool_timeout: int = 60
    judge_provider: Optional[str] = None   # None = model graders return UNKNOWN
    judge_model: Optional[str] = None

    @staticmethod
    def load(data_dir: Optional[Path] = None, start: Optional[Path] = None) -> "Settings":
        s = Settings()
        cfg = _find_config(start or Path.cwd())
        if cfg:
            raw = json.loads(cfg.read_text(encoding="utf-8"))
            s.data_dir = Path(raw.get("data_dir", s.data_dir))
            s.provider = raw.get("provider", s.provider)
            s.model = raw.get("model", s.model)
            s.tool_timeout = int(raw.get("tool_timeout", s.tool_timeout))
            s.judge_provider = raw.get("judge_provider", s.judge_provider)
            s.judge_model = raw.get("judge_model", s.judge_model)
            if not s.data_dir.is_absolute():
                s.data_dir = cfg.parent / s.data_dir

        s.provider = os.environ.get("AGENTWB_PROVIDER", s.provider)
        s.model = os.environ.get("AGENTWB_MODEL", s.model)
        s.judge_provider = os.environ.get("AGENTWB_JUDGE_PROVIDER", s.judge_provider)
        s.judge_model = os.environ.get("AGENTWB_JUDGE_MODEL", s.judge_model)
        if os.environ.get("AGENTWB_DATA_DIR"):
            s.data_dir = Path(os.environ["AGENTWB_DATA_DIR"])
        if data_dir is not None:
            s.data_dir = Path(data_dir)

        s.data_dir.mkdir(parents=True, exist_ok=True)
        return s


def _find_config(start: Path) -> Optional[Path]:
    for d in [start, *start.parents]:
        for name in CONFIG_NAMES:
            p = d / name
            if p.is_file():
                return p
    return None
