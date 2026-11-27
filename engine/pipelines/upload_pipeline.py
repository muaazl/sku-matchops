import csv
import logging
import re
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from engine.core import config

logger = logging.getLogger("matchops.engine.uploader")


def _target_page_size(sku_count: int) -> str | None:
    """Returns the Ant Design pagination size ('20', '50', '100'), or None to keep default 10."""
    if sku_count <= 10:
        return None
    if sku_count <= 20:
        return "20"
    if sku_count <= 50:
        return "50"
    return "100"


def process_upload(skus, outlet_id, job_id, progress_callback, is_cancelled, has_more_jobs=False):
    """Executes the Playwright bulk upload automation for the provided SKUs."""
    if not skus:
        raise ValueError("No SKUs provided for upload")

    staging_dir = Path(config.STAGING_DIR)
    staging_dir.mkdir(parents=True, exist_ok=True)
    temp_csv_path = staging_dir / f"temp_upload_{job_id}.csv"
    try:
        keys = skus[0].keys()
        with temp_csv_path.open("w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(skus)
    except Exception as e:
        logger.error(f"Error creating temp CSV: {e}")
        raise RuntimeError(f"Error creating temp CSV: {e}")

    progress_callback("uploading", 5.0)
    results = []

    try:
        with sync_playwright() as p:
            if is_cancelled():
                return {"results": results}

            try:
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else context.new_page()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to Chrome on port 9222. Ensure Chrome is running with --remote-debugging-port=9222. Error: {e}"
                )

            base_url = "https://uni-portal.pickme.lk/"
            target_url = f"https://uni-portal.pickme.lk/#/t0001/food-portal/food-manage/restaurants/{outlet_id}/manage-skus"

            logger.info(f"[JOB {job_id}] Navigating to base URL to check login state...")
            page.goto(base_url)
            page.wait_for_load_state("networkidle")
            time.sleep(1)

            if "login" in page.url:
                logger.info(
                    f"[JOB {job_id}] Login required. Waiting up to 5 minutes for user to log in manually..."
                )
                try:
                    page.wait_for_url("**/#/t0001/food-portal**", timeout=300000)
                    logger.info(f"[JOB {job_id}] Login detected successfully!")
                    time.sleep(2)
                except PlaywrightTimeoutError:
                    raise RuntimeError(
                        "Timeout waiting for manual login. Please run the upload again when you are ready."
                    )

            logger.info(f"[JOB {job_id}] Navigating to target outlet URL: {target_url}")
            page.goto(target_url)
            page.wait_for_load_state("networkidle")
            time.sleep(1)

            try:
                page.get_by_text("SKU Bulk Upload").click(timeout=5000)
            except Exception:
                page.locator(
                    "xpath=/html/body/div[1]/div/div/div/div/main/section/div[2]/div/div/div/div[1]/div[1]/div/div[4]/div/span"
                ).click()

            time.sleep(1.5)
            progress_callback("uploading", 20.0)

            file_input = page.locator("input[type='file']")
            if file_input.count() > 0:
                file_input.set_input_files(temp_csv_path)
            else:
                with page.expect_file_chooser() as fc_info:
                    page.locator(
                        "xpath=/html/body/div[1]/div/div/div/div/main/section/div[2]/div/div/div/div[2]/div/div/div/div/div[2]/span[2]/div[1]/span/button/span[2]"
                    ).click()
                file_chooser = fc_info.value
                file_chooser.set_files(temp_csv_path)

            time.sleep(2)
            progress_callback("uploading", 40.0)

            target_size = _target_page_size(len(skus))
            if target_size:
                target_label = f"{target_size} / page"
                try:
                    size_selector = page.locator(
                        ".ant-select-selector:has(span.ant-select-selection-item[title*='/ page']):visible"
                    ).first
                    size_selector.wait_for(state="visible", timeout=8000)

                    if target_label not in (size_selector.inner_text() or ""):
                        size_selector.scroll_into_view_if_needed()
                        size_selector.click(timeout=5000)

                        option = (
                            page.locator(".ant-select-dropdown:visible .ant-select-item-option")
                            .filter(
                                has_text=re.compile(
                                    rf"^\s*{target_size}\s*/\s*page\s*$", re.IGNORECASE
                                )
                            )
                            .first
                        )
                        option.wait_for(state="visible", timeout=5000)
                        option.click(timeout=5000)
                        page.locator(
                            f"span.ant-select-selection-item[title='{target_label}']:visible"
                        ).first.wait_for(state="visible", timeout=5000)
                        time.sleep(1.0)
                        logger.info(f"[JOB {job_id}] Changed page size to '{target_label}'.")
                except Exception as e:
                    logger.warning(
                        f"[JOB {job_id}] Could not change page size to '{target_label}': {e}"
                    )

            current_page = 1
            pages_processed = 0

            while True:
                if is_cancelled():
                    break

                try:
                    page.locator("thead th").first.locator("svg").first.click(timeout=10000)
                    time.sleep(0.5)
                    page.get_by_text("Select all to update").click()
                    time.sleep(0.5)

                    page.get_by_role("button").filter(
                        has_text=re.compile(r"Bulk Action", re.IGNORECASE)
                    ).click()

                    toast_locator = page.locator("xpath=/html/body/div[5]/div[1]/div/div")
                    toast_locator.wait_for(state="visible", timeout=30000)
                    toast_text = toast_locator.text_content().strip()
                    logger.info(f"[JOB {job_id}] Page {current_page} toast: '{toast_text}'")

                    try:
                        toast_locator.wait_for(state="hidden", timeout=10000)
                    except PlaywrightTimeoutError:
                        pass
                    pages_processed += 1
                except Exception as e:
                    logger.error(f"[JOB {job_id}] Error on page {current_page}: {e}")

                next_button = page.locator("li.ant-pagination-next")
                if next_button.count() == 0:
                    is_disabled = True
                else:
                    class_attr = next_button.get_attribute("class") or ""
                    is_disabled = (
                        "ant-pagination-disabled" in class_attr
                        or next_button.get_attribute("aria-disabled") == "true"
                    )

                if is_disabled:
                    break
                next_button.click()
                current_page += 1
                time.sleep(1.5)
                pct = min(90.0, 40.0 + (pages_processed * 5.0))
                progress_callback("uploading", pct)

            progress_callback("uploading", 100.0)

            try:
                page.locator(
                    "xpath=/html/body/div[1]/div/div/div/div/main/section/div[2]/div/div/div/div[2]/div/div/div/div/div[2]/span[2]/div[2]/div/div/span[2]/button"
                ).click(timeout=5000)
                logger.info(f"[JOB {job_id}] Clicked 'Clear CSV' button.")
                time.sleep(1)

                page.locator(
                    "xpath=/html/body/div[5]/div/div[2]/div/div[1]/div/div[2]/div/div[2]/button[2]"
                ).click(timeout=5000)
                logger.info(f"[JOB {job_id}] Clicked 'Remove' on the confirmation modal.")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"[JOB {job_id}] Could not clear CSV (button or modal failed): {e}")

            if not has_more_jobs:
                try:
                    logger.info(f"[JOB {job_id}] No more jobs in queue. Navigating to home page.")
                    page.goto("https://uni-portal.pickme.lk/#/t0001/food-portal")
                    page.wait_for_load_state("networkidle")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"[JOB {job_id}] Could not navigate to home page: {e}")

            for sku in skus:
                results.append(
                    {
                        "status": "High (Uploaded)",
                        "sku_name": sku.get(
                            "Name", sku.get("sku_name", sku.get("name", "Unknown"))
                        ),
                    }
                )

    finally:
        if temp_csv_path.exists():
            temp_csv_path.unlink()

    return {"results": results, "pages_processed": pages_processed}
