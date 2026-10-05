#!/usr/bin/env python3
"""
db_query.py - Universal database query & export tool for sku-matchops.

Features:
  1. Job Summary: Sheet Name, Total SKUs, High Conf SKUs, Match Rate, etc.
  2. SKU Details: Sku name, matched catalog, confidences, categories, keywords, BT, source.
  3. Table Explorer: Query any table with custom columns, filters (WHERE), sorting, limits.
  4. Raw SQL: Run any custom SELECT query directly from CLI or interactive prompt.
  5. Schema Inspector: View all tables, column definitions, and row counts.
  6. Professional Formatting (-fmt / --formatted): Google Sheets & Excel styling with tailored column widths, sticky headers, zebra striping, and auto-filters.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
DEFAULT_DB_PATH = DATA_DIR / "sku-matchops.db"
EXPORTS_DIR = DATA_DIR / "exports"


def get_db_path() -> Path:
    env_db = os.getenv("DB_PATH")
    if env_db and Path(env_db).exists():
        return Path(env_db)
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    raise FileNotFoundError(f"Database not found at {DEFAULT_DB_PATH}")


def get_connection(readonly: bool = True) -> sqlite3.Connection:
    db_path = get_db_path()
    if readonly:
        uri_path = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def sanitize_filename(name: str) -> str:
    """Removes characters forbidden in filenames on Windows/Linux."""
    return re.sub(r'[<>:"/\\|?*\']', "", str(name)).strip()


def parse_job_ids(raw_input: str) -> list:
    """Safely extracts clean job IDs, stripping quotes, spaces, hashes, brackets, etc."""
    if not raw_input:
        return []
    cleaned = str(raw_input).strip().strip("\"'[](){}")
    tokens = cleaned.replace(";", ",").split(",")
    job_ids = []
    for t in tokens:
        token = t.strip().strip("\"'# ")
        if token:
            job_ids.append(token)
    return job_ids


def sanitize_sheet_name(name: str, fallback: str, used_names: set) -> str:
    """Sanitizes sheet names to conform with Excel 31-char limit and invalid chars."""
    raw = (name or "").strip() or fallback
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", raw).strip() or fallback
    candidate = cleaned[:31].strip() or fallback[:31].strip()

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    counter = 2
    while True:
        suffix = f"_{counter}"
        truncated = candidate[: 31 - len(suffix)] + suffix
        if truncated not in used_names:
            used_names.add(truncated)
            return truncated
        counter += 1


def format_conf(val):
    if val is None or val == "":
        return ""
    try:
        return round(float(val), 4)
    except (ValueError, TypeError):
        return val


def parse_generic_keywords(gk_val) -> str:
    if not gk_val:
        return ""
    if isinstance(gk_val, list):
        return ", ".join(str(x) for x in gk_val if str(x).strip())
    try:
        parsed = json.loads(gk_val)
        if isinstance(parsed, list):
            return ", ".join(str(x) for x in parsed if str(x).strip())
        return str(parsed)
    except Exception:
        return str(gk_val).strip()


def print_table(headers: list, rows: list, max_col_width: int = 50, max_rows: int = 100):
    """Pretty prints a table to terminal without external dependencies."""
    if not headers:
        print("[No data]")
        return

    # Convert all cells to strings and truncate if too long
    str_rows = []
    for r in rows[:max_rows]:
        row_cells = []
        for h in headers:
            v = r.get(h, "") if isinstance(r, dict) else r[h]
            val_str = "" if v is None else str(v).replace("\n", " ").replace("\r", "")
            if len(val_str) > max_col_width:
                val_str = val_str[: max_col_width - 3] + "..."
            row_cells.append(val_str)
        str_rows.append(row_cells)

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format dividers
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for row in str_rows:
        line = "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        print(line)
    print(sep)

    if len(rows) > max_rows:
        print(f"(Showing first {max_rows} of {len(rows)} rows)")
    else:
        print(f"Total: {len(rows)} row(s)")


def apply_sheet_formatting(
    ws, headers: list, col_widths_override: dict = None, freeze_panes: str = "A2"
):
    """
    Applies professional styling for Google Sheets & Excel:
      - Dark Navy Slate header with bold white text (28pt row height)
      - Sticky/frozen header row
      - Pre-applied auto-filter dropdowns across all columns
      - Subtle zebra striping (#F8FAFC vs #FFFFFF, 20pt row height)
      - Clean thin cell borders (#E2E8F0)
      - Gridlines explicitly enabled
      - Content-aware column widths tailored for readability (no congestion)
    """
    try:
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter

        header_font = Font(name="Google Sans", size=11, bold=True, color="000000")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        data_font = Font(name="Google Sans", size=10)
        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

        thin_side = Side(style="thin", color="E2E8F0")
        cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        align_left = Alignment(horizontal="left", vertical="center")
        align_center = Alignment(horizontal="center", vertical="center")
        align_right = Alignment(horizontal="right", vertical="center")

        ws.views.sheetView[0].showGridLines = True
        if freeze_panes:
            ws.freeze_panes = freeze_panes
        ws.row_dimensions[1].height = 28

        # Style header row (bold text, no colored background)
        for c_idx in range(1, len(headers) + 1):
            c = ws.cell(row=1, column=c_idx)
            c.font = header_font
            c.alignment = header_align
            c.border = cell_border

        # Alignments based on header name
        alignments = {}
        for c_idx, h in enumerate(headers, 1):
            h_lower = str(h).lower()
            if any(
                k in h_lower
                for k in ["conf", "rate", "score", "status", "source", "duration", "id"]
            ):
                alignments[c_idx] = align_center
            elif any(k in h_lower for k in ["total", "count", "items"]):
                alignments[c_idx] = align_right
            else:
                alignments[c_idx] = align_left

        # Style data rows
        max_r = ws.max_row
        for row_idx, row_cells in enumerate(
            ws.iter_rows(min_row=2, max_row=max_r, max_col=len(headers)), start=2
        ):
            ws.row_dimensions[row_idx].height = 20
            fill = zebra_fill if row_idx % 2 == 1 else white_fill
            for c_idx, cell in enumerate(row_cells, start=1):
                cell.font = data_font
                cell.fill = fill
                cell.border = cell_border
                cell.alignment = alignments.get(c_idx, align_left)

        if max_r >= 1:
            ws.auto_filter.ref = ws.dimensions

        # Column widths
        for c_idx, h in enumerate(headers, 1):
            col_letter = get_column_letter(c_idx)
            if col_widths_override and c_idx in col_widths_override:
                ws.column_dimensions[col_letter].width = col_widths_override[c_idx]
            elif col_widths_override and h in col_widths_override:
                ws.column_dimensions[col_letter].width = col_widths_override[h]
            else:
                h_lower = str(h).lower()
                if any(k in h_lower for k in ["sku", "catalog", "name"]):
                    w = 38
                elif any(k in h_lower for k in ["keyword", "notes", "description"]):
                    w = 36
                elif any(k in h_lower for k in ["category", "categories", "rule"]):
                    w = 28
                elif any(k in h_lower for k in ["type", "sheet", "store"]):
                    w = 24
                elif any(k in h_lower for k in ["date", "time", "completed", "started", "at"]):
                    w = 22
                elif any(k in h_lower for k in ["conf", "rate", "status", "duration"]):
                    w = 18
                elif "id" in h_lower:
                    w = 12
                else:
                    w = max(len(str(h)) + 4, 14)
                ws.column_dimensions[col_letter].width = w
    except Exception as e:
        print(f"[Note] Sheet formatting error ({e})")


def export_data(
    data: list,
    base_name: str,
    export_csv: bool = True,
    export_xlsx: bool = True,
    sheet_name: str = "Sheet1",
    formatted: bool = False,
):
    """Exports list of dictionaries to CSV and/or XLSX in exports directory."""
    if not data:
        print("[Notice] No data to export.")
        return

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_base = sanitize_filename(base_name) or "export"
    fieldnames = list(data[0].keys())

    if export_csv:
        csv_file = EXPORTS_DIR / f"{clean_base}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved CSV:  {csv_file}")

    if export_xlsx:
        if formatted:
            try:
                import openpyxl

                out_name = (
                    f"{clean_base}_formatted"
                    if not clean_base.endswith("_formatted")
                    else clean_base
                )
                xlsx_file = EXPORTS_DIR / f"{out_name}.xlsx"
                wb = openpyxl.Workbook()
                ws = wb.active
                valid_sheet = sanitize_sheet_name(sheet_name, "Data", set())
                ws.title = valid_sheet
                ws.append(fieldnames)
                for row_dict in data:
                    ws.append(
                        ["" if row_dict.get(h) is None else row_dict.get(h) for h in fieldnames]
                    )
                apply_sheet_formatting(ws, fieldnames)
                wb.save(xlsx_file)
                print(f"Saved Formatted XLSX (Google Sheets ready): {xlsx_file}")
            except Exception as e:
                print(f"[Note] Formatted XLSX export failed ({e})")
        else:
            try:
                import pandas as pd

                xlsx_file = EXPORTS_DIR / f"{clean_base}.xlsx"
                df = pd.DataFrame(data)
                valid_sheet = sanitize_sheet_name(sheet_name, "Data", set())
                df.to_excel(xlsx_file, index=False, sheet_name=valid_sheet)
                print(f"Saved XLSX: {xlsx_file}")
            except Exception as e:
                print(f"[Note] XLSX export skipped ({e})")


# ==============================================================================
# FEATURE 1: Job Summary (Sheet Name, Total SKUs, High Conf SKUs, Match Rate, etc.)
# ==============================================================================


def query_job_summaries(conn: sqlite3.Connection, job_ids: list) -> list:
    """Fetches high-level metrics for given job numbers."""
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in job_ids)

    query = f"""
    SELECT
        j.id AS job_id,
        COALESCE(NULLIF(j.sheet_name, ''), NULLIF(j.target_sheet, ''), 'Job ' || j.id) AS sheet_name,
        COALESCE(j.total_items, (SELECT COUNT(*) FROM processed_skus p WHERE p.batch_id = j.id)) AS total_skus,
        COALESCE(j.high_conf, (SELECT COUNT(*) FROM processed_skus p WHERE p.batch_id = j.id AND p.confidence >= 0.8)) AS high_conf_skus,
        COALESCE(j.med_conf, 0) AS med_conf_skus,
        COALESCE(j.low_conf, 0) AS low_conf_skus,
        ROUND(COALESCE(j.match_rate, 0.0), 2) AS match_rate,
        j.status AS status,
        ROUND(COALESCE(j.duration_minutes, 0.0), 2) AS duration_min,
        j.domain AS domain,
        j.completed_at AS completed_at
    FROM jobs j
    WHERE j.id IN ({placeholders})
    ORDER BY CAST(j.id AS INTEGER) ASC
    """
    cursor.execute(query, [str(j).strip().lstrip("#") for j in job_ids])
    rows = cursor.fetchall()
    return [dict(r) for r in rows]


# ==============================================================================
# FEATURE 2: SKU Match Details (Sku name, matched catalog, confidences, etc.)
# ==============================================================================


def query_job_skus(conn: sqlite3.Connection, job_id: str) -> list:
    """Fetches detailed SKU rows for a given job."""
    cursor = conn.cursor()
    job_str = str(job_id).strip().lstrip("#")

    query = """
    SELECT
        sku_name,
        matched_catalog_name,
        match_score,
        confidence,
        bt_confidence,
        category,
        region,
        gk_json,
        bt,
        match_source
    FROM processed_skus
    WHERE batch_id = ?
       OR batch_id IN (SELECT batch_id FROM jobs WHERE id = ?)
    ORDER BY ROWID ASC
    """
    cursor.execute(query, (job_str, job_str))
    rows = cursor.fetchall()

    results = []
    for r in rows:
        raw_source = (r["match_source"] or "").strip()
        source_lower = raw_source.lower()
        score = r["match_score"]
        conf = r["confidence"]
        bt_conf = r["bt_confidence"]

        if "matcher" in source_lower:
            matcher_conf = score if score is not None else conf
            classifier_conf = ""
        elif "classifier" in source_lower:
            classifier_conf = conf if conf is not None else bt_conf
            matcher_conf = score if (score is not None and r["matched_catalog_name"]) else ""
        else:
            matcher_conf = score if score is not None else ""
            classifier_conf = conf if conf is not None else bt_conf

        category_val = r["category"] if r["category"] else (r["region"] or "")

        results.append(
            {
                "Sku name": r["sku_name"] or "",
                "matched catalog": r["matched_catalog_name"] or "",
                "matcher confidence": format_conf(matcher_conf),
                "classifier confidence": format_conf(classifier_conf),
                "categories": category_val,
                "generic keywords": parse_generic_keywords(r["gk_json"]),
                "basic type": r["bt"] or "",
                "source (matcher or classifier)": raw_source,
            }
        )
    return results


def export_multi_job_skus(
    conn: sqlite3.Connection, job_ids: list, formatted: bool = False, out_name: str = None
):
    """Exports SKU details for multiple jobs into separate CSVs and multi-tab XLSX."""
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in job_ids)
    cursor.execute(
        f"SELECT id, COALESCE(NULLIF(sheet_name, ''), NULLIF(target_sheet, ''), 'Job ' || id) AS sheet_name FROM jobs WHERE id IN ({placeholders})",
        [str(j).strip().lstrip("#") for j in job_ids],
    )
    sheet_map = {str(r["id"]): r["sheet_name"] for r in cursor.fetchall()}

    all_jobs_data = {}
    total_records = 0

    for j_id in job_ids:
        j_str = str(j_id).strip().lstrip("#")
        sheet_label = sheet_map.get(j_str, f"Job {j_str}")
        skus = query_job_skus(conn, j_str)
        all_jobs_data[j_str] = (sheet_label, skus)
        total_records += len(skus)
        print(f"  -> Job #{j_str} ({sheet_label}): {len(skus)} SKUs found")

    if total_records == 0:
        print("[Warning] No SKU records found for the given jobs.")
        return

    summary_name = sanitize_filename(out_name) if out_name else "_".join(job_ids[:5])
    if not out_name and len(job_ids) > 5:
        summary_name += f"_and_{len(job_ids) - 5}_more"

    # Save CSVs
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for j_str, (sheet_label, skus) in all_jobs_data.items():
        if skus:
            csv_path = EXPORTS_DIR / f"job_{j_str}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(skus[0].keys()))
                writer.writeheader()
                writer.writerows(skus)
            print(f"Saved CSV:  {csv_path}")
            for r in skus:
                c_row = {"Job ID": j_str, "Sheet Name": sheet_label}
                c_row.update(r)
                combined_rows.append(c_row)

    if len(all_jobs_data) > 1 and combined_rows:
        comb_csv_path = EXPORTS_DIR / f"jobs_skus_{summary_name}_combined.csv"
        with open(comb_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"Saved Master Combined CSV: {comb_csv_path}")

    # Save multi-tab Excel
    try:
        if formatted:
            import openpyxl

            xlsx_path = EXPORTS_DIR / f"jobs_skus_{summary_name}_formatted.xlsx"
            wb = openpyxl.Workbook()
            wb.remove(wb.active)  # remove default sheet

            # 1. Overview & Stores Index Tab if multiple jobs
            if len(job_ids) > 1:
                summaries = query_job_summaries(conn, job_ids)
                if summaries:
                    ws_sum = wb.create_sheet(title="Overview & Stores")
                    sum_headers = [
                        "Job ID",
                        "Store / Sheet Name",
                        "Total SKUs",
                        "High Conf SKUs",
                        "Med Conf SKUs",
                        "Low Conf SKUs",
                        "Match Rate (%)",
                        "Status",
                        "Duration (min)",
                        "Completed At",
                    ]
                    ws_sum.append(sum_headers)
                    for s in summaries:
                        rate_val = (
                            f"{s['match_rate']:.2f}%"
                            if s.get("match_rate") is not None
                            else "0.00%"
                        )
                        dur_val = (
                            f"{s['duration_min']:.2f}"
                            if s.get("duration_min") is not None
                            else "0.00"
                        )
                        ws_sum.append(
                            [
                                s.get("job_id"),
                                s.get("sheet_name"),
                                s.get("total_skus"),
                                s.get("high_conf_skus"),
                                s.get("med_conf_skus"),
                                s.get("low_conf_skus"),
                                rate_val,
                                s.get("status"),
                                dur_val,
                                str(s.get("completed_at") or ""),
                            ]
                        )
                    apply_sheet_formatting(
                        ws_sum,
                        sum_headers,
                        col_widths_override={
                            1: 12,
                            2: 38,
                            3: 14,
                            4: 16,
                            5: 16,
                            6: 16,
                            7: 16,
                            8: 14,
                            9: 16,
                            10: 22,
                        },
                    )

            # 2. Individual store tabs
            used_sheets = {"Overview & Stores"}
            sku_col_widths = {1: 38, 2: 38, 3: 18, 4: 18, 5: 28, 6: 36, 7: 22, 8: 20}
            for j_str, (sheet_label, skus) in all_jobs_data.items():
                tab_name = sanitize_sheet_name(sheet_label, f"Job {j_str}", used_sheets)
                ws = wb.create_sheet(title=tab_name)
                headers = [
                    "Sku name",
                    "matched catalog",
                    "matcher confidence",
                    "classifier confidence",
                    "categories",
                    "generic keywords",
                    "basic type",
                    "source (matcher or classifier)",
                ]
                ws.append(headers)
                for row in skus:
                    ws.append([row.get(h, "") for h in headers])
                apply_sheet_formatting(
                    ws, headers, col_widths_override=sku_col_widths, freeze_panes="A2"
                )

            wb.save(xlsx_path)
            print(f"\nSaved Formatted Multi-Tab XLSX (Google Sheets ready): {xlsx_path}")
        else:
            import pandas as pd

            xlsx_path = EXPORTS_DIR / f"jobs_skus_{summary_name}.xlsx"
            used_sheets = set()
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                for j_str, (sheet_label, skus) in all_jobs_data.items():
                    tab_name = sanitize_sheet_name(sheet_label, f"Job {j_str}", used_sheets)
                    if skus:
                        df = pd.DataFrame(skus)
                    else:
                        df = pd.DataFrame(
                            columns=[
                                "Sku name",
                                "matched catalog",
                                "matcher confidence",
                                "classifier confidence",
                                "categories",
                                "generic keywords",
                                "basic type",
                                "source (matcher or classifier)",
                            ]
                        )
                    df.to_excel(writer, sheet_name=tab_name, index=False)
            print(f"\nCombined Multi-Tab XLSX: {xlsx_path}")
    except Exception as e:
        print(f"[Error creating Excel] {e}")


# ==============================================================================
# FEATURE 3: General Table Query & Raw SQL
# ==============================================================================


def query_table(
    conn: sqlite3.Connection,
    table: str,
    columns: str = "*",
    where: str = None,
    order: str = None,
    limit: int = None,
) -> list:
    """Queries any table with optional column selection, filter, sort, and limit."""
    cursor = conn.cursor()
    col_str = columns if columns and columns.strip() else "*"
    sql = f"SELECT {col_str} FROM {table}"
    if where and where.strip():
        sql += f" WHERE {where}"
    if order and order.strip():
        sql += f" ORDER BY {order}"
    if limit and int(limit) > 0:
        sql += f" LIMIT {int(limit)}"

    cursor.execute(sql)
    rows = cursor.fetchall()
    return [dict(r) for r in rows]


def run_raw_sql(conn: sqlite3.Connection, sql: str) -> list:
    """Executes a custom SQL statement and returns results as dicts."""
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    return [dict(r) for r in rows]


def list_tables_and_schema(conn: sqlite3.Connection):
    """Prints all tables, their row counts, and column definitions."""
    cursor = conn.cursor()
    tables = [
        r[0]
        for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    summary = []
    for t in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
        summary.append(
            {
                "Table Name": t,
                "Row Count": count,
                "Columns": ", ".join(cols[:8]) + ("..." if len(cols) > 8 else ""),
            }
        )

    print_table(["Table Name", "Row Count", "Columns"], summary, max_col_width=60, max_rows=100)


# ==============================================================================
# INTERACTIVE CLI MENU
# ==============================================================================


def interactive_menu(conn: sqlite3.Connection):
    while True:
        print("\n" + "=" * 60)
        print("  SKU-MATCHOPS DATABASE QUERY TOOL")
        print("=" * 60)
        print("  [1] Job Summary (Sheet Name, Total SKUs, High Conf SKUs, etc.)")
        print("  [2] Job SKU Details (All SKU matches, confidences, categories)")
        print("  [3] Query Any Table (Select table, columns, WHERE filter)")
        print("  [4] Run Custom SQL Query")
        print("  [5] Show All Tables & Schemas")
        print("  [0] Exit")
        print("-" * 60)

        choice = input("Select an option [0-5]: ").strip()
        if choice in ("0", "exit", "q"):
            print("Exiting. Bye!")
            break

        if choice == "1":
            raw = input("\nEnter Job Number(s) (comma-separated, e.g. 414, 415, 420): ").strip()
            if not raw:
                continue
            job_ids = parse_job_ids(raw)
            if not job_ids:
                print("[Error] No valid job IDs entered.")
                continue
            data = query_job_summaries(conn, job_ids)
            if data:
                print("\nJob Summaries:")
                print_table(list(data[0].keys()), data, max_col_width=35)
                save_prompt = input("\nExport to CSV and XLSX? [Y/n]: ").strip().lower()
                if save_prompt in ("", "y", "yes"):
                    fmt_prompt = (
                        input("Apply professional styling (Google Sheets / Excel ready)? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                    summary_name = "_".join(job_ids[:4])
                    if len(job_ids) > 4:
                        summary_name += f"_and_{len(job_ids) - 4}_more"
                    export_data(
                        data,
                        f"job_summary_{summary_name}",
                        export_csv=True,
                        export_xlsx=True,
                        sheet_name="Job Summary",
                        formatted=fmt_prompt in ("", "y", "yes"),
                    )
            else:
                print("[Notice] No records found for the given jobs.")

        elif choice == "2":
            raw = input("\nEnter Job Number(s) (comma-separated, e.g. 414, 415, 420): ").strip()
            if not raw:
                continue
            job_ids = parse_job_ids(raw)
            if not job_ids:
                print("[Error] No valid job IDs entered.")
                continue
            fmt_prompt = (
                input("Apply professional styling (Google Sheets / Excel ready)? [Y/n]: ")
                .strip()
                .lower()
            )
            print(f"\nExtracting SKU details for {len(job_ids)} job(s)...")
            export_multi_job_skus(conn, job_ids, formatted=fmt_prompt in ("", "y", "yes"))

        elif choice == "3":
            cursor = conn.cursor()
            tables = [
                r[0]
                for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            print("\nAvailable tables: " + ", ".join(tables))
            table_name = input("Table name: ").strip()
            if table_name not in tables:
                print(f"[Error] Table '{table_name}' does not exist.")
                continue

            cols_info = [
                r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            print(f"Columns: {', '.join(cols_info)}")

            cols_input = input("Columns to select (Enter for *): ").strip() or "*"
            where_input = input(
                "WHERE clause (e.g. status='completed', or press Enter to skip): "
            ).strip()
            order_input = input("ORDER BY clause (or press Enter to skip): ").strip()
            limit_input = input("LIMIT (default 50): ").strip() or "50"

            try:
                data = query_table(
                    conn,
                    table_name,
                    columns=cols_input,
                    where=where_input,
                    order=order_input,
                    limit=int(limit_input),
                )
                if data:
                    print(f"\nResults from {table_name}:")
                    print_table(list(data[0].keys()), data, max_col_width=40, max_rows=50)
                    save_prompt = input("\nExport to CSV and XLSX? [Y/n]: ").strip().lower()
                    if save_prompt in ("", "y", "yes"):
                        fmt_prompt = (
                            input(
                                "Apply professional styling (Google Sheets / Excel ready)? [Y/n]: "
                            )
                            .strip()
                            .lower()
                        )
                        export_data(
                            data,
                            f"table_{table_name}",
                            export_csv=True,
                            export_xlsx=True,
                            sheet_name=table_name,
                            formatted=fmt_prompt in ("", "y", "yes"),
                        )
                else:
                    print("[Notice] No matching rows found.")
            except Exception as e:
                print(f"[Query Error] {e}")

        elif choice == "4":
            print("\nEnter custom SQL query (must start with SELECT):")
            sql = input("> ").strip()
            if not sql:
                continue
            if not sql.lower().startswith("select"):
                print("[Error] Only SELECT queries are permitted.")
                continue
            try:
                data = run_raw_sql(conn, sql)
                if data:
                    print("\nQuery Results:")
                    print_table(list(data[0].keys()), data, max_col_width=40, max_rows=50)
                    save_prompt = input("\nExport to CSV and XLSX? [Y/n]: ").strip().lower()
                    if save_prompt in ("", "y", "yes"):
                        fmt_prompt = (
                            input(
                                "Apply professional styling (Google Sheets / Excel ready)? [Y/n]: "
                            )
                            .strip()
                            .lower()
                        )
                        export_data(
                            data,
                            "custom_query_results",
                            export_csv=True,
                            export_xlsx=True,
                            formatted=fmt_prompt in ("", "y", "yes"),
                        )
                else:
                    print("[Notice] Query returned 0 rows.")
            except Exception as e:
                print(f"[SQL Error] {e}")

        elif choice == "5":
            print("\nDatabase Schema & Tables:")
            list_tables_and_schema(conn)

        input("\nPress Enter to continue...")


# ==============================================================================
# MAIN / CLI PARSER
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="High-level database query & export tool for sku-matchops.db"
    )
    # Mode flags
    parser.add_argument(
        "-js",
        "--job-summary",
        help="Comma-separated job IDs to get sheet name, total SKUs, high conf SKUs, etc.",
    )
    parser.add_argument(
        "-skus", "--job-skus", help="Comma-separated job IDs to extract full SKU match details"
    )
    parser.add_argument(
        "-t",
        "--table",
        help="Table name to query (e.g. jobs, processed_skus, rules, catalog_items)",
    )
    parser.add_argument(
        "-c", "--columns", default="*", help="Columns to select (comma-separated, default: *)"
    )
    parser.add_argument("-w", "--where", help="WHERE filter clause (e.g. \"status='completed'\")")
    parser.add_argument("-o", "--order", help='ORDER BY clause (e.g. "id DESC")')
    parser.add_argument(
        "-l", "--limit", type=int, default=100, help="Max rows to return (default: 100)"
    )
    parser.add_argument("--sql", help="Raw SELECT SQL statement to execute")
    parser.add_argument(
        "--tables", action="store_true", help="List all tables, row counts, and columns"
    )

    # Output & Styling flags
    parser.add_argument("--csv", action="store_true", help="Export results to CSV in ./exports/")
    parser.add_argument("--xlsx", action="store_true", help="Export results to XLSX in ./exports/")
    parser.add_argument(
        "-fmt",
        "--formatted",
        action="store_true",
        help="Apply professional Google Sheets / Excel styling (proper column widths, dark header, frozen panes, zebra striping, borders, auto-filters)",
    )
    parser.add_argument("--out", help="Custom base filename for export (saved under ./exports/)")

    args = parser.parse_args()

    try:
        conn = get_connection(readonly=True)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    # If no CLI arguments given, launch interactive menu
    if len(sys.argv) == 1:
        interactive_menu(conn)
        conn.close()
        return

    # CLI 1: Show tables
    if args.tables:
        list_tables_and_schema(conn)
        conn.close()
        return

    # CLI 2: Job Summary
    if args.job_summary:
        job_ids = parse_job_ids(args.job_summary)
        data = query_job_summaries(conn, job_ids)
        if data:
            print("\nJob Summaries:")
            print_table(list(data[0].keys()), data, max_col_width=40)
            base_name = args.out or f"job_summary_{'_'.join(job_ids[:4])}"
            # Honor explicit flags: if only --csv, do not export xlsx
            if args.csv and not (args.xlsx or args.formatted):
                do_csv, do_xlsx = True, False
            elif (args.xlsx or args.formatted) and not args.csv:
                do_csv, do_xlsx = False, True
            else:
                do_csv, do_xlsx = True, True

            export_data(
                data,
                base_name,
                export_csv=do_csv,
                export_xlsx=do_xlsx,
                sheet_name="Job Summary",
                formatted=args.formatted,
            )
        else:
            print("[Notice] No records found for the given jobs.")
        conn.close()
        return

    # CLI 3: Job SKUs
    if args.job_skus:
        job_ids = parse_job_ids(args.job_skus)
        print(f"\nExtracting SKU details for {len(job_ids)} job(s)...")
        export_multi_job_skus(conn, job_ids, formatted=args.formatted, out_name=args.out)
        conn.close()
        return

    # CLI 4: Table query
    if args.table:
        try:
            data = query_table(
                conn,
                args.table,
                columns=args.columns,
                where=args.where,
                order=args.order,
                limit=args.limit,
            )
            if data:
                print(f"\nQuery from table '{args.table}':")
                print_table(list(data[0].keys()), data, max_col_width=40, max_rows=args.limit)
                if args.csv or args.xlsx or args.formatted or args.out:
                    base_name = args.out or f"table_{args.table}"
                    export_data(
                        data,
                        base_name,
                        export_csv=args.csv or (not args.xlsx and not args.formatted),
                        export_xlsx=args.xlsx or args.formatted,
                        sheet_name=args.table,
                        formatted=args.formatted,
                    )
            else:
                print(f"[Notice] 0 rows returned from table '{args.table}'.")
        except Exception as e:
            print(f"[Error] {e}")
        conn.close()
        return

    # CLI 5: Raw SQL
    if args.sql:
        if not args.sql.strip().lower().startswith("select"):
            print("[Error] Only SELECT queries are permitted.")
            conn.close()
            sys.exit(1)
        try:
            data = run_raw_sql(conn, args.sql)
            if data:
                print("\nSQL Results:")
                print_table(list(data[0].keys()), data, max_col_width=40, max_rows=args.limit)
                if args.csv or args.xlsx or args.formatted or args.out:
                    base_name = args.out or "sql_results"
                    export_data(
                        data,
                        base_name,
                        export_csv=args.csv or (not args.xlsx and not args.formatted),
                        export_xlsx=args.xlsx or args.formatted,
                        formatted=args.formatted,
                    )
            else:
                print("[Notice] 0 rows returned from SQL query.")
        except Exception as e:
            print(f"[Error] {e}")
        conn.close()
        return

    conn.close()


if __name__ == "__main__":
    main()
