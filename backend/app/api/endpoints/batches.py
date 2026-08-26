import csv
import io
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form

from backend.app.core.db import get_db_connection
from backend.app.schemas.models import (
    BaseRequest,
    BatchResponse,
    MerchantFetchRequest,
    SKUItem,
)
from backend.app.services.worker import enqueue_job

router = APIRouter()

@router.post("/batches")
async def create_batch(
    domain: str = Form(...),
    created_by: str = Form(...),
    task: str = Form("pipeline"),
    file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db_connection)
):
    if task not in ("pipeline", "matcher", "classifier"):
        raise HTTPException(status_code=400, detail="Invalid task type.")
    
    filename = file.filename or "Job"
    if filename:
        for ext in ('.csv', '.tsv', '.txt'):
            if filename.lower().endswith(ext):
                filename = filename[:-len(ext)]
                break
    content = await file.read()
    
    # Parse CSV content
    try:
        csv_file = io.StringIO(content.decode('utf-8', errors='ignore'))
        reader = csv.DictReader(csv_file)
        skus = []
        
        name_col = next((f for f in (reader.fieldnames or []) if f.lower().strip() in ('name', 'sku_name', 'sku', 'title')), None)
        price_col = next((f for f in (reader.fieldnames or []) if f.lower().strip() in ('price', 'cost', 'mrp')), None)
        desc_col = next((f for f in (reader.fieldnames or []) if f.lower().strip() in ('description', 'desc')), None)
        cat_col = next((f for f in (reader.fieldnames or []) if f.lower().strip() in ('category', 'cat', 'type')), None)

        if not name_col and reader.fieldnames:
            name_col = reader.fieldnames[0]

        for row in reader:
            if not name_col:
                continue
            name_val = row.get(name_col, "").strip() if name_col else ""
            if not name_val:
                continue
            
            price_val = 0.0
            if price_col:
                try:
                    price_val = float(row.get(price_col, 0))
                except ValueError:
                    pass
            desc_val = row.get(desc_col, "").strip() if desc_col else ""
            cat_val = row.get(cat_col, "").strip() if cat_col else ""
            
            skus.append(SKUItem(name=name_val, price=price_val, description=desc_val, category=cat_val))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if not skus:
        raise HTTPException(status_code=400, detail="CSV file has no valid SKUs")

    # Enqueue a pipeline job
    request = BaseRequest(
        skus=skus,
        domain=domain,
        callback_url="",
        sheet_name=filename # filename serves as target sheet name
    )
    
    res = enqueue_job(request, task=task)
    job_id = res["job_id"]
    
    # Insert batch entry
    db.execute(
        """
        INSERT INTO batches (id, source, filename, domain, status, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, 'upload', filename, domain, 'queued', created_by)
    )
    db.commit()
    
    return {"batch_id": job_id, "job_id": job_id}

@router.post("/merchant-fetch")
def merchant_fetch(request: MerchantFetchRequest, db: sqlite3.Connection = Depends(get_db_connection)):
    # Generate some mock SKUs for the merchant fetch based on domain
    if request.domain == 'food':
        skus = [
            SKUItem(name="Organic Avocado", price=1.99, description="Fresh organic avocado", category="Produce"),
            SKUItem(name="Whole Milk 1L", price=2.49, description="Whole pasteurized milk", category="Dairy"),
            SKUItem(name="Whole Wheat Bread", price=3.19, description="Fresh baked whole wheat bread", category="Bakery"),
            SKUItem(name="Greek Yogurt 500g", price=4.50, description="Plain Greek yogurt", category="Dairy"),
            SKUItem(name="Spaghetti 500g", price=1.20, description="Durum wheat semolina pasta", category="Pantry"),
        ]
    else:
        skus = [
            SKUItem(name="Wireless Mouse", price=25.00, description="Optical wireless mouse", category="Electronics"),
            SKUItem(name="Cotton T-Shirt M", price=15.99, description="100% cotton black t-shirt", category="Apparel"),
            SKUItem(name="Stainless Water Bottle", price=18.50, description="Vacuum insulated bottle", category="Home"),
            SKUItem(name="Running Shoes US9", price=85.00, description="Lightweight sports sneakers", category="Footwear"),
            SKUItem(name="Notebook A5", price=4.99, description="Ruled journal notebook", category="Stationery"),
        ]
        
    req = BaseRequest(
        skus=skus,
        domain=request.domain,
        callback_url="",
        sheet_name=f"Merchant {request.merchant_id}"
    )
    
    res = enqueue_job(req, task=request.task)
    job_id = res["job_id"]
    
    db.execute(
        """
        INSERT INTO batches (id, source, merchant_id, domain, status, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, 'merchant', request.merchant_id, request.domain, 'queued', 'system')
    )
    db.commit()
    
    
    return {"batch_id": job_id, "job_id": job_id}

@router.get("/batches/{id}", response_model=BatchResponse)
def get_batch(id: str, db: sqlite3.Connection = Depends(get_db_connection)):
    row = db.execute("SELECT * FROM batches WHERE id = ?", (id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Batch not found")
    return dict(row)
