"""Shared CSV writing helper for experiment dataclass records."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Sequence, Tuple, Union

Column = Union[str, Tuple[str, str]]  # field_name, or (header, field_name)


def write_dataclass_csv(
    records: Iterable,
    path: Path,
    columns: Sequence[Column],
    formatters: Mapping[str, Callable[[object], object]] | None = None,
) -> None:
    """Write dataclass records to CSV.

    columns: field names, or (header, field_name) pairs when the header differs.
    formatters: per-field-name formatter functions applied before writing.
    """
    fmt = formatters or {}
    pairs: List[Tuple[str, str]] = [
        (c, c) if isinstance(c, str) else c for c in columns
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header for header, _ in pairs])
        for record in records:
            d = asdict(record)
            writer.writerow([
                fmt[field](d[field]) if field in fmt else d[field]
                for _, field in pairs
            ])
