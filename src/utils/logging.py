from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


def append_metrics_csv(path: str, row: Dict[str, float | int]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
