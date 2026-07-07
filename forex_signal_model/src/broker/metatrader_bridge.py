from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class BrokerConfig:
    # Common approach: run an MT4/MT5 EA that polls a local endpoint or reads a file.
    # This Python side writes a JSON signal to disk.
    signals_file: str = "signals_out.json"


class FileSignalBroker:
    """Writes the latest signal to a file.

    You then implement an MT4/MT5 EA that reads this JSON and executes trades.
    """

    def __init__(self, cfg: BrokerConfig):
        self.cfg = cfg

    def publish(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["ts"] = int(time.time())
        tmp = self.cfg.signals_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self.cfg.signals_file)


class HttpSignalBroker:
    """Optionally POST the signal to a locally hosted endpoint."""

    def __init__(self, endpoint_url: str, timeout_s: float = 5.0):
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s

    def publish(self, payload: Dict[str, Any]) -> None:
        requests.post(self.endpoint_url, json=payload, timeout=self.timeout_s)

