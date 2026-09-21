"""Portable paths for exported records and log messages."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_text(value: str) -> str:
    """Remove machine-specific home paths from a serialized description."""
    value = value.replace(str(PROJECT_ROOT) + "/", "")
    value = value.replace(str(PROJECT_ROOT), ".")
    value = value.replace(sys.executable, "python")
    value = re.sub(r"/home/[^/\s\"']+/", "external/", value)
    value = re.sub(r"/Users/[^/\s\"']+/", "external/", value)
    return value


def portable_metadata(value):
    """Normalize JSON/CSV metadata; leave numerical objects untouched."""
    if isinstance(value, (str, Path)):
        return portable_text(str(value))
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    return value


class PortableFormatter(logging.Formatter):
    """Format logs without recording the local account's absolute paths."""

    def format(self, record):
        return portable_text(super().format(record))
