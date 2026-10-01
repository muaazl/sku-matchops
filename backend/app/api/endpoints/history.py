import csv
import io
import sqlite3
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.app.core.db import get_db_connection
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


@router.get("/processed-skus", response_model=List[ProcessedSkuResponse])
def get_history(
    response: Response,
    batch_id: Optional[str] = None,
    domain: Optional[str] = None,
    min_confidence: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku_name: Optional[str] = None,
    bt: Optional[str] = None,
    gk: Optional[str] = None,
    match_source: Optional[str] = None,
    page: int = 1,
    limit: Optional[int] = 50,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    query = "SELECT * FROM processed_skus WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM processed_skus WHERE 1=1"
    params = []
    
    if batch_id:
        query += " AND batch_id = ?"
        count_query += " AND batch_id = ?"
        params.append(batch_id)
    if domain:
        query += " AND domain = ?"
        count_query += " AND domain = ?"
        params.append(domain)
    if min_confidence is not None:
        query += " AND confidence >= ?"
        count_query += " AND confidence >= ?"
        params.append(min_confidence)
    if date_from:
        query += " AND created_at >= ?"
        count_query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        count_query += " AND created_at <= ?"
        params.append(date_to)
    if sku_name:
        query += " AND sku_name LIKE ?"
        count_query += " AND sku_name LIKE ?"
        params.append(f"%{sku_name}%")
    if bt:
        query += " AND (basic_type LIKE ? OR bt LIKE ?)"
        count_query += " AND (basic_type LIKE ? OR bt LIKE ?)"
        params.extend([f"%{bt}%", f"%{bt}%"])
    if gk:
        query += " AND (generic_keywords LIKE ? OR gk_json LIKE ? OR gk LIKE ?)"
        count_query += " AND (generic_keywords LIKE ? OR gk_json LIKE ? OR gk LIKE ?)"
        params.extend([f"%{gk}%", f"%{gk}%", f"%{gk}%"])
    if match_source:
        query += " AND match_source LIKE ?"
        count_query += " AND match_source LIKE ?"
        params.append(f"%{match_source}%")
        
    total_count = db.execute(count_query, params).fetchone()[0]
    response.headers["X-Total-Count"] = str(total_count)

    query += " ORDER BY created_at DESC"
    if limit is not None and limit > 0:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, (page - 1) * limit])
    
    rows = db.execute(query, params).fetchall()
    return [dict(row) for row in rows]

@router.get("/processed-skus/export")
def export_history(
    ids: Optional[str] = None,
    format: Literal["csv", "xlsx"] = "csv",
    db: sqlite3.Connection = Depends(get_db_connection)
):
    query = "SELECT * FROM processed_skus"
    params = []

    if ids:
        id_list = [i.strip() for i in ids.split(',') if i.strip()]
        if id_list:
            placeholders = ','.join('?' for _ in id_list)
            query += f" WHERE id IN ({placeholders})"
            params.extend(id_list)
    else:
        # No explicit id filter: cap the export instead of dumping the entire table.
        query += " ORDER BY created_at DESC LIMIT 50000"

    rows = db.execute(query, params).fetchall()

    if format == 'csv':
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _sanitize_csv_cell(v) for k, v in dict(row).items()})

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=export.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="XLSX export not fully implemented yet. Please use format=csv.")

@router.get("/processed-skus/{id}", response_model=ProcessedSkuResponse)
def get_processed_sku(id: str, db: sqlite3.Connection = Depends(get_db_connection)):
    row = db.execute("SELECT * FROM processed_skus WHERE id = ?", (id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="SKU not found")
    return dict(row)
