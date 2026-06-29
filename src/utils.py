from typing import Any


def describe_signal(signal: Any) -> str:
    """Return a readable description for a raw signal value."""
    return str(signal).strip().lower()
