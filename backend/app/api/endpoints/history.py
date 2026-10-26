import csv
import io
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import Connection, text

from backend.app.core.db import get_db_connection
from backend.app.core.security import Principal, get_principal
from backend.app.schemas.models import ProcessedSkuResponse

router = APIRouter()

# Prefixes that spreadsheet apps (Excel/Sheets) interpret as the start of a formula.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _sanitize_csv_cell(value):
    """Neutralizes CSV/formula injection: a leading =, +, -, or @ makes Excel/Sheets
    treat the cell as a formula to execute on open. Prefix with a tab to defuse it
    while keeping the visible value unchanged for a human reading the export."""
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return "\t" + value
    return value


@router.get("/processed-skus", response_model=list[ProcessedSkuResponse])
def get_history(
    response: Response,
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
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    where = "WHERE tenant_id = :tenant_id"
    params = {"tenant_id": principal.tenant_id}

    if batch_id:
        where += " AND batch_id = :batch_id"
        params["batch_id"] = batch_id
    if domain:
        where += " AND domain = :domain"
        params["domain"] = domain
    if min_confidence is not None:
        where += " AND confidence >= :min_confidence"
        params["min_confidence"] = min_confidence
    if date_from:
        where += " AND created_at >= CAST(:date_from AS timestamptz)"
        params["date_from"] = date_from
    if date_to:
        where += " AND created_at <= CAST(:date_to AS timestamptz)"
        params["date_to"] = date_to
    if sku_name:
        where += " AND sku_name LIKE :sku_name"
        params["sku_name"] = f"%{sku_name}%"
    if bt:
        where += " AND bt LIKE :bt"
        params["bt"] = f"%{bt}%"
    if gk:
        where += " AND gk_json LIKE :gk"
        params["gk"] = f"%{gk}%"
    if match_source:
        where += " AND match_source LIKE :match_source"
        params["match_source"] = f"%{match_source}%"

    total_count = db.execute(text(f"SELECT COUNT(*) FROM processed_skus {where}"), params).scalar()
    response.headers["X-Total-Count"] = str(total_count)

    query = f"SELECT * FROM processed_skus {where} ORDER BY created_at DESC"
    if limit is not None and limit > 0:
        query += " LIMIT :limit OFFSET :offset"
        params.update(limit=limit, offset=(page - 1) * limit)

    rows = db.execute(text(query), params).mappings().fetchall()
    return [dict(row) for row in rows]


@router.get("/processed-skus/export")
def export_history(
    ids: str | None = None,
    format: Literal["csv", "xlsx"] = "csv",
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    query = "SELECT * FROM processed_skus WHERE tenant_id = :tenant_id"
    params = {"tenant_id": principal.tenant_id}

    if ids:
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        if id_list:
            query += " AND id = ANY(:ids)"
            params["ids"] = id_list
    else:
        # No explicit id filter: cap the export instead of dumping the entire table.
        query += " ORDER BY created_at DESC LIMIT 50000"

    rows = db.execute(text(query), params).mappings().fetchall()

    if format == "csv":
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _sanitize_csv_cell(v) for k, v in dict(row).items()})

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=export.csv"},
        )
    else:
        raise HTTPException(
            status_code=400, detail="XLSX export not fully implemented yet. Please use format=csv."
        )


@router.get("/processed-skus/{id}", response_model=ProcessedSkuResponse)
def get_processed_sku(
    id: str,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(get_principal),
):
    row = (
        db.execute(
            text("SELECT * FROM processed_skus WHERE id = :id AND tenant_id = :tenant_id"),
            {"id": id, "tenant_id": principal.tenant_id},
        )
        .mappings()
        .fetchone()
    )
    if not row:
        raise HTTPException(status_code=404, detail="SKU not found")
    return dict(row)
