"""Processed SKU history querying, detail lookup, and formula-safe CSV streaming export."""

import csv
import io
from collections.abc import Iterator
from typing import Any

from backend.core.exceptions import ResourceNotFoundError
from backend.repositories import sku_repository

# Prefixes that spreadsheet apps (Excel/Sheets) interpret as the start of a formula.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")


def sanitize_csv_cell(value: Any) -> Any:
    """Neutralizes CSV/formula injection by prefixing risky leading operators with a tab."""
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return "\t" + value
    return value


def list_processed_skus(
    *,
    batch_id: str | None = None,
    domain: str | None = None,
    min_confidence: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sku_name: str | None = None,
    bt: str | None = None,
    gk: str | None = None,
    match_source: str | None = None,
    page: int = 1,
    limit: int | None = 50,
) -> tuple[int, list[dict[str, Any]]]:
    """Queries paginated processed SKU audit rows matching optional filters."""
    return sku_repository.count_and_fetch_processed_skus(
        batch_id=batch_id,
        domain=domain,
        min_confidence=min_confidence,
        date_from=date_from,
        date_to=date_to,
        sku_name=sku_name,
        bt=bt,
        gk=gk,
        match_source=match_source,
        page=page,
        limit=limit,
    )


def fetch_processed_sku_detail(sku_id: str) -> dict[str, Any]:
    """Fetches a single processed SKU record or raises ResourceNotFoundError."""
    row = sku_repository.fetch_processed_sku_by_id(sku_id)
    if not row:
        raise ResourceNotFoundError("SKU not found")
    return row


def generate_csv_chunks(ids: str | None = None, db: Any = None) -> Iterator[str]:
    """Generates formula-sanitized CSV chunks for streaming export."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()][:1000] if ids else None
    buf = io.StringIO()
    writer: csv.DictWriter | None = None

    for partition in sku_repository.stream_processed_skus_partitions(ids=id_list, db=db):
        for row_dict in partition:
            if writer is None:
                writer = csv.DictWriter(buf, fieldnames=row_dict.keys())
                writer.writeheader()
            writer.writerow({k: sanitize_csv_cell(v) for k, v in row_dict.items()})
        chunk = buf.getvalue()
        if chunk:
            yield chunk
            buf.seek(0)
            buf.truncate(0)
