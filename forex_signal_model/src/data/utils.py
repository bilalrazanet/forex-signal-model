from __future__ import annotations


def parse_interval_minutes(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Unsupported interval format: {interval}")
