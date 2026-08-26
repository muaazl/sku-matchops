import csv
import io
import sqlite3
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.app.core.db import get_db_connection
from backend.app.schemas.models import ProcessedSkuResponse

router = APIRouter()

@router.get("/processed-skus", response_model=List[ProcessedSkuResponse])
def get_history(
    batch_id: Optional[str] = None,
    domain: Optional[str] = None,
    min_confidence: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    limit: Optional[int] = 50,
    db: sqlite3.Connection = Depends(get_db_connection)
):
    query = "SELECT * FROM processed_skus WHERE 1=1"
    params = []
    
    if batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    if min_confidence is not None:
        query += " AND confidence >= ?"
        params.append(min_confidence)
    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to)
        
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
        
    rows = db.execute(query, params).fetchall()
    
    if format == 'csv':
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=dict(rows[0]).keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
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
