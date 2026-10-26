/**
 * SKU MatchOps – Google Sheets Integration
 * ==========================================
 * Setup:
 *  1. Extensions > Apps Script → paste this file.
 *  2. Update CONFIG below with your Cloudflare Tunnel URL and API key.
 *  3. Deploy as a Web App (Deploy → New Deployment → Web App,
 *     Execute as: Me, Who has access: Anyone).
 *     Paste the Web App URL into CONFIG.CALLBACK_URL below.
 *  4. Refresh the Sheet — a new "SKU MatchOps" menu will appear.
 *
 * How async works:
 *  - Matcher runs synchronously (fast, no timeout risk).
 *  - Classifier fires a job to the server and returns immediately.
 *    The server POSTs results back via doPost() when done.
 */

const CONFIG = {
  // The tunnel URL is now read dynamically from the _config tab.
  // Use getBaseUrl() instead of CONFIG.BASE_URL directly.
  // Fallback only used if _config tab has no URL yet.
  BASE_URL_FALLBACK: "",

  // Google Sheet ID of the catalog sheet where _config tab lives.
  CATALOG_SHEET_ID: "17wPPEUu0_KbgOPtjDy4UVS8fh76D1ZmQlB5FZMw2JEo",

  // Must match MATCHOPS_API_KEY in your .env (default: "bismillah")
  API_KEY: "meilimasterkey",

  // Your Apps Script Web App URL (for async callback).
  // Get this from: Deploy → Manage Deployments → copy the Web App URL.
  CALLBACK_URL: "https://script.google.com/macros/s/AKfycbzSY6pZ_8GAKO6ThLOLeh7MkuFDNQzjUrVm9SXZoa2CfFwsOJRTQx6W15UwzmRV0voONA/exec",

  // Column name in your active sheet that holds the SKU / dish name
  NAME_COL: "Name",

  // Column name for price (optional)
  PRICE_COL: "Price",

  // Column name for description (optional)
  DESC_COL: "Description",

  // Column name for seller category (optional)
  CATEGORY_COL: "Category",

  // Output column names for Matcher
  MATCHER_OUT_COLS: {
    matched_name:  "Matched Catalog Name",
    score:         "Match Score",
    status:        "Match Status",
    logic_notes:   "Logic Notes",
    rules_applied: "Rules Applied",
  },

  // Output column names for Classifier
  CLASSIFIER_OUT_COLS: {
    suggested_bt:      "Predicted Basic Type",
    bt_confidence:     "BT Confidence",
    bt_status:         "BT Status",
    suggested_gk:      "Predicted Generic Keywords",
    gk_confidence:     "GK Confidence",
    gk_status:         "GK Status",
    suggested_region:  "Predicted Region/Category",
    region_confidence: "Region/Cat Confidence",
    region_status:     "Region/Cat Status",
    rules_applied:     "Rules Applied",
  },

  // Output column names for Pipeline (Matcher + Classifier combined)
  PIPELINE_OUT_COLS: {
    suggested_gk:      "Predicted Generic Keywords",
    suggested_bt:      "Predicted Basic Type",
    suggested_region:  "Predicted Region/Category",
    pipeline_source:   "Source",
  },
};

// ─── Queue & Tunnel Helpers ──────────────────────────────────────────────────

/**
 * Returns the _queue sheet from the central catalog spreadsheet.
 * Creates it with headers if it doesn't exist.
 */
function _getOrCreateQueueSheet() {
  const ss = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
  let sheet = ss.getSheetByName("_queue");
  if (!sheet) {
    sheet = ss.insertSheet("_queue");
    sheet.getRange("A1:G1").setValues([["Job ID", "Spreadsheet ID", "Sheet Name", "Domain", "Task", "Status", "Created At"]]);
    sheet.getRange("A1:G1").setFontWeight("bold");
    sheet.setColumnWidth(1, 280);
    sheet.setColumnWidth(2, 280);
    sheet.setColumnWidth(3, 160);
    sheet.setColumnWidth(4, 100);
    sheet.setColumnWidth(5, 100);
    sheet.setColumnWidth(6, 120);
    sheet.setColumnWidth(7, 180);
  }
  return sheet;
}

/**
 * Returns the _config sheet from the catalog spreadsheet.
 * Creates it with headers if it doesn't exist.
 */
function _getOrCreateConfigSheet() {
  const ss = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
  let sheet = ss.getSheetByName("_config");
  if (!sheet) {
    sheet = ss.insertSheet("_config");
    sheet.getRange("A1:B1").setValues([["Key", "Value"]]);
    sheet.getRange("A2:B2").setValues([["tunnel_url", ""]]);
    sheet.getRange("A3:B3").setValues([["updated_at", ""]]);
    sheet.setColumnWidth(1, 120);
    sheet.setColumnWidth(2, 400);
    sheet.getRange("A1:B1").setFontWeight("bold");
  }
  return sheet;
}

/**
 * Reads the latest tunnel URL from the _config tab on the catalog sheet.
 * Falls back to CONFIG.BASE_URL_FALLBACK if not set.
 */
function getBaseUrl() {
  try {
    const sheet = _getOrCreateConfigSheet();
    const url = sheet.getRange("B2").getValue();
    if (url && String(url).trim() !== "") {
      return String(url).trim();
    }
  } catch (e) {
    Logger.log("[MatchOps] Failed to read tunnel URL from _config: " + e.message);
  }
  return CONFIG.BASE_URL_FALLBACK;
}

/**
 * Writes a tunnel URL to the _config tab on the catalog sheet.
 */
function _writeTunnelUrl(url) {
  const sheet = _getOrCreateConfigSheet();
  sheet.getRange("B2").setValue(url);
  sheet.getRange("B3").setValue(new Date().toISOString());
  Logger.log("[MatchOps] Tunnel URL updated to: " + url);
}

/**
 * Shows the current tunnel URL in a dialog.
 */
function showCurrentUrl() {
  const url = getBaseUrl();
  const ui = SpreadsheetApp.getUi();
  if (url) {
    ui.alert("Current Tunnel URL", "The backend is at:\n\n" + url + "\n\nThis was auto-set by the start_tunnel script.", ui.ButtonSet.OK);
  } else {
    ui.alert("No URL Set", "No tunnel URL found in the _config tab.\nRun start_tunnel.ps1 or use 'Set URL Manually'.", ui.ButtonSet.OK);
  }
}

/**
 * Prompts the user to manually enter a tunnel URL.
 */
function setUrlManually() {
  const ui = SpreadsheetApp.getUi();
  const result = ui.prompt(
    "Set Tunnel URL",
    "Paste your Cloudflare Tunnel URL (e.g. https://xyz.trycloudflare.com):",
    ui.ButtonSet.OK_CANCEL
  );
  if (result.getSelectedButton() === ui.Button.OK) {
    const url = result.getResponseText().trim();
    if (url.startsWith("https://")) {
      _writeTunnelUrl(url);
      ui.alert("URL Updated", "Tunnel URL set to:\n" + url, ui.ButtonSet.OK);
    } else {
      ui.alert("Invalid URL", "The URL must start with https://", ui.ButtonSet.OK);
    }
  }
}

// // ─── Menu ────────────────────────────────────────────────────────────────────

// function onOpen() {
//   SpreadsheetApp.getUi()
//     .createMenu("SKU MatchOps")
//     .addItem("Matcher - Food",       "runMatcherFood")
//     .addItem("Matcher - Market",     "runMatcherMarket")
//     .addSeparator()
//     .addItem("Classifier - Food",    "runClassifierFood")
//     .addItem("Classifier - Market",  "runClassifierMarket")
//     .addSeparator()
//     .addItem("Pipeline - Food",      "runPipelineFood")
//     .addItem("Pipeline - Market",    "runPipelineMarket")
//     .addSeparator()
//     .addItem("🔗 Show Current URL",  "showCurrentUrl")
//     .addItem("✏️ Set URL Manually",  "setUrlManually")
//     .addItem("Setup Instructions",   "showInstructions")
//     .addToUi();
// }

function runMatcherFood()      { runMatching("food",   "matcher"); }
function runMatcherMarket()    { runMatching("market", "matcher"); }
function runClassifierFood()   { runMatching("food",   "classifier"); }
function runClassifierMarket() { runMatching("market", "classifier"); }
function runPipelineFood()     { runMatching("food",   "pipeline"); }
function runPipelineMarket()   { runMatching("market", "pipeline"); }

// ─── Main trigger ─────────────────────────────────────────────────────────────

function runMatching(domain, task) {
  const ss    = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getActiveSheet();
  const ui    = SpreadsheetApp.getUi();

  // 1. Read headers
  const lastCol = sheet.getLastColumn();
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) { ui.alert("Sheet is empty."); return; }

  const initialHeaders = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map(String);

  // Delete any existing output columns first to reset
  const allOutputColNames = new Set([
    ...Object.values(CONFIG.MATCHER_OUT_COLS),
    ...Object.values(CONFIG.CLASSIFIER_OUT_COLS),
    ...Object.values(CONFIG.PIPELINE_OUT_COLS)
  ]);

  // Loop backwards to avoid index shift issues when deleting
  for (let c = lastCol; c >= 1; c--) {
    const headerVal = initialHeaders[c - 1];
    if (allOutputColNames.has(headerVal)) {
      sheet.deleteColumn(c);
    }
  }

  // Re-read headers after deletion
  const cleanLastCol = sheet.getLastColumn();
  const headers      = sheet.getRange(1, 1, 1, cleanLastCol).getValues()[0].map(String);
  const nameIdx      = headers.findIndex(h => h.toLowerCase() === CONFIG.NAME_COL.toLowerCase());
  const priceIdx     = headers.findIndex(h => h.toLowerCase() === CONFIG.PRICE_COL.toLowerCase());
  const descIdx      = headers.findIndex(h => h.toLowerCase() === CONFIG.DESC_COL.toLowerCase());
  const catIdx       = headers.findIndex(h => h.toLowerCase() === CONFIG.CATEGORY_COL.toLowerCase());

  if (nameIdx === -1) {
    ui.alert(`Could not find a "${CONFIG.NAME_COL}" column in this sheet.`);
    return;
  }

  // 2. Ensure output columns exist
  const outCols   = task === "classifier" ? CONFIG.CLASSIFIER_OUT_COLS
                 : task === "pipeline"   ? CONFIG.PIPELINE_OUT_COLS
                 : CONFIG.MATCHER_OUT_COLS;
  const outIdxMap = {};

  Object.entries(outCols).forEach(([key, colName]) => {
    let idx = headers.findIndex(h => h === colName);
    if (idx === -1) {
      sheet.getRange(1, cleanLastCol + Object.keys(outIdxMap).length + 1).setValue(colName);
      idx = cleanLastCol + Object.keys(outIdxMap).length;
    }
    outIdxMap[key] = idx;
  });

  // Refresh after additions
  const finalLastCol   = sheet.getLastColumn();
  const updatedHeaders = sheet.getRange(1, 1, 1, finalLastCol).getValues()[0].map(String);
  Object.entries(outCols).forEach(([key, colName]) => {
    outIdxMap[key] = updatedHeaders.findIndex(h => h === colName);
  });

  // 3. Read data rows
  const dataRange = sheet.getRange(2, 1, lastRow - 1, finalLastCol);
  const data      = dataRange.getValues();

  // 4. Build SKU list
  const skus = data.map(row => ({
    name:        String(row[nameIdx] || ""),
    price:       priceIdx !== -1 ? (parseFloat(row[priceIdx]) || 0) : 0,
    description: descIdx !== -1 ? String(row[descIdx] || "") : "",
    category:    catIdx !== -1 ? String(row[catIdx] || "") : "",
  })).filter(s => s.name.trim() !== "");

  if (skus.length === 0) { ui.alert("No SKU names found."); return; }

  // 5. Build payload & queue job in the central CATALOG sheet
  const jobId = Utilities.getUuid();
  const queueSheet = _getOrCreateQueueSheet();
  queueSheet.appendRow([jobId, ss.getId(), sheet.getName(), domain, task, "PENDING", new Date().toISOString()]);

  // 6. Attempt direct connection if baseUrl is configured; fallback to local queue mode
  const baseUrl = getBaseUrl();
  if (baseUrl) {
    try {
      const payload = {
        job_id:         jobId,
        skus:           skus,
        domain:         domain,
        callback_url:   CONFIG.CALLBACK_URL,
        spreadsheet_id: ss.getId(),
        sheet_name:     sheet.getName(),
      };
      const endpoint = task === "matcher" ? "match" : task === "classifier" ? "classify" : "pipeline";
      const response = UrlFetchApp.fetch(`${baseUrl}/${endpoint}`, {
        method:             "post",
        contentType:        "application/json",
        headers:            { "X-API-Key": CONFIG.API_KEY },
        payload:            JSON.stringify(payload),
        muteHttpExceptions: true,
      });

      if (response.getResponseCode() === 200) {
        ss.toast(
          `Job submitted directly! Processing ${skus.length} SKUs in background. Results will appear in "${sheet.getName()}".`,
          "✅ SKU MatchOps"
        );
        return;
      }
    } catch (e) {
      Logger.log("[MatchOps] Direct tunnel fetch failed, queued for local worker: " + e.message);
    }
  }

  // Pure Local Queue mode (Zero-Tunnel)
  ss.toast(
    `Job queued! [${task.toUpperCase()}] for tab "${sheet.getName()}". Local MatchOps will process and update rows automatically.`,
    "⏳ SKU MatchOps",
    10
  );
}

// ─── Query receiver (doGet) ──────────────────────────────────────────────────
// Allows the local Python sync script to query available tabs or pending queue jobs.

function doGet(e) {
  try {
    const action = (e && e.parameter && e.parameter.action) || "status";
    const ssId = (e && e.parameter && e.parameter.spreadsheet_id) || CONFIG.CATALOG_SHEET_ID;
    const ss = SpreadsheetApp.openById(ssId);

    // 1. List sheets/tabs in the spreadsheet
    if (action === "list_sheets") {
      const sheets = ss.getSheets()
        .map(s => s.getName())
        .filter(n => !n.startsWith("_")); // Exclude internal tabs like _config and _queue
      return ContentService.createTextOutput(JSON.stringify({ status: "ok", sheets: sheets }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // 2. Fetch all pending jobs from the _queue tab in the CATALOG sheet
    if (action === "get_queue") {
      const catalogSs = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
      const queueSheet = catalogSs.getSheetByName("_queue");
      if (!queueSheet || queueSheet.getLastRow() < 2) {
        return ContentService.createTextOutput(JSON.stringify({ status: "ok", pending: [] }))
          .setMimeType(ContentService.MimeType.JSON);
      }
      const data = queueSheet.getDataRange().getValues();
      const pending = [];
      for (let r = 1; r < data.length; r++) {
        const row = data[r];
        if (row.length >= 7 && row[5] === "PENDING") {
          pending.push({
            row_index: r + 1,
            job_id: String(row[0]),
            spreadsheet_id: String(row[1]),
            sheet_name: String(row[2]),
            domain: String(row[3]),
            task: String(row[4]),
            status: String(row[5]),
            created_at: String(row[6]),
          });
        } else if (row[4] === "PENDING") {
          pending.push({
            row_index: r + 1,
            job_id: String(row[0]),
            spreadsheet_id: ssId,
            sheet_name: String(row[1]),
            domain: String(row[2]),
            task: String(row[3]),
            status: String(row[4]),
            created_at: String(row[5]),
          });
        }
      }
      return ContentService.createTextOutput(JSON.stringify({ status: "ok", pending: pending }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // 3. Read data from a specific tab directly
    if (action === "get_data") {
      const targetSheetName = e.parameter.sheet_name;
      const sheet = targetSheetName ? ss.getSheetByName(targetSheetName) : ss.getActiveSheet();
      if (!sheet) {
        return ContentService.createTextOutput(JSON.stringify({ status: "error", message: `Sheet "${targetSheetName}" not found.` }))
          .setMimeType(ContentService.MimeType.JSON);
      }
      const data = sheet.getDataRange().getValues();
      return ContentService.createTextOutput(JSON.stringify({
        status: "ok",
        sheet_name: sheet.getName(),
        data: data
      })).setMimeType(ContentService.MimeType.JSON);
    }

    return ContentService.createTextOutput(JSON.stringify({
      status: "ok",
      service: "SKU MatchOps Apps Script",
      spreadsheet_id: ssId
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

// ─── Callback receiver (doPost) ───────────────────────────────────────────────
// The local Python server calls this URL when the async job is complete.

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);

    // Handle tunnel URL update from start_tunnel.ps1
    if (body.type === "tunnel_update" && body.url) {
      _writeTunnelUrl(body.url);
      return ContentService.createTextOutput(JSON.stringify({ status: "ok", url: body.url }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Handle queue status update from local worker (always targets central CATALOG sheet)
    if (body.type === "queue_update" && body.job_id) {
      const catalogSs = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
      const queueSheet = catalogSs.getSheetByName("_queue");
      if (queueSheet) {
        const qData = queueSheet.getDataRange().getValues();
        for (let r = 1; r < qData.length; r++) {
          if (String(qData[r][0]) === String(body.job_id)) {
            const statusColIdx = qData[0].length >= 7 ? 6 : 5;
            queueSheet.getRange(r + 1, statusColIdx).setValue(body.status);
            break;
          }
        }
      }
      return ContentService.createTextOutput(JSON.stringify({ status: "ok" }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Error case
    if (body.error) {
      Logger.log(`[MatchOps] Job ${body.job_id} failed: ${body.error}`);
      return ContentService.createTextOutput(JSON.stringify({ status: "error_logged" }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const ss      = SpreadsheetApp.openById(body.spreadsheet_id || CONFIG.CATALOG_SHEET_ID);
    const sheet   = ss.getSheetByName(body.sheet_name);
    const task    = body.task || "classifier";
    const results = body.results || [];

    if (!sheet) {
      Logger.log(`[MatchOps] Sheet "${body.sheet_name}" not found.`);
      return ContentService.createTextOutput(JSON.stringify({ status: "sheet_not_found" }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const lastCol      = sheet.getLastColumn();
    const lastRow      = sheet.getLastRow();
    const headers      = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map(String);
    const nameIdx      = headers.findIndex(h => h.toLowerCase() === CONFIG.NAME_COL.toLowerCase());
    const outCols      = task === "classifier" ? CONFIG.CLASSIFIER_OUT_COLS
                       : task === "pipeline"   ? CONFIG.PIPELINE_OUT_COLS
                       : CONFIG.MATCHER_OUT_COLS;
    const outIdxMap    = {};
    Object.entries(outCols).forEach(([key, colName]) => {
      outIdxMap[key] = headers.findIndex(h => h === colName);
    });

    const dataRange = sheet.getRange(2, 1, lastRow - 1, lastCol);
    const data      = dataRange.getValues();

    _writeResults(data, dataRange, results, task, nameIdx, outIdxMap);

    // Mark job as COMPLETED in _queue in the central CATALOG sheet if job_id is present
    if (body.job_id) {
      const catalogSs = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
      const queueSheet = catalogSs.getSheetByName("_queue");
      if (queueSheet) {
        const qData = queueSheet.getDataRange().getValues();
        for (let r = 1; r < qData.length; r++) {
          if (String(qData[r][0]) === String(body.job_id)) {
            const statusColIdx = qData[0].length >= 7 ? 6 : 5;
            queueSheet.getRange(r + 1, statusColIdx).setValue("COMPLETED");
            break;
          }
        }
      }
    }

    ss.toast(
      `Job finished! Successfully updated ${results.length} rows in "${body.sheet_name}".`,
      "✅ SKU MatchOps"
    );

    Logger.log(`[MatchOps] Job ${body.job_id} — wrote ${results.length} results to "${body.sheet_name}".`);
    return ContentService.createTextOutput(JSON.stringify({ status: "ok" }))
      .setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    Logger.log(`[MatchOps] doPost error: ${err.message}`);
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

// ─── Shared write logic ────────────────────────────────────────────────────────

function _writeResults(data, dataRange, results, task, nameIdx, outIdxMap) {
  const backgrounds = dataRange.getBackgrounds();
  let resultIdx = 0;

  for (let i = 0; i < data.length; i++) {
    const name = String(data[i][nameIdx] || "").trim();
    if (!name) {
      // Clear values and backgrounds for output columns of rows with empty names
      Object.values(outIdxMap).forEach(colIdx => {
        if (colIdx !== -1 && colIdx < data[i].length) {
          data[i][colIdx] = "";
          backgrounds[i][colIdx] = null;
        }
      });
      continue;
    }

    const res = results[resultIdx++];
    if (!res) continue;

    if (task === "matcher") {
      data[i][outIdxMap.matched_name] = res.matched_catalog_name || "";
      data[i][outIdxMap.score]        = res.score ?? "";
      data[i][outIdxMap.status]       = res.status || "";
      data[i][outIdxMap.logic_notes]  = res.logic_notes || "";
      data[i][outIdxMap.rules_applied]= res.rules_applied || "";

      const isLow = !res.status || String(res.status).toLowerCase().includes("low") || String(res.status).toLowerCase().includes("rejected");
      const hl    = isLow ? "#FFF2CC" : null;
      backgrounds[i][outIdxMap.matched_name] = hl;
      backgrounds[i][outIdxMap.score]        = hl;
      backgrounds[i][outIdxMap.status]       = hl;
      backgrounds[i][outIdxMap.logic_notes]  = hl;
      backgrounds[i][outIdxMap.rules_applied]= hl;

    } else if (task === "classifier") {
      data[i][outIdxMap.suggested_bt]      = res.suggested_bt || "";
      data[i][outIdxMap.bt_confidence]     = res.bt_confidence ?? "";
      data[i][outIdxMap.bt_status]         = res.bt_status || "";
      data[i][outIdxMap.suggested_gk]      = res.suggested_gk || "";
      data[i][outIdxMap.gk_confidence]     = res.gk_confidence ?? "";
      data[i][outIdxMap.gk_status]         = res.gk_status || "";
      data[i][outIdxMap.suggested_region]  = res.suggested_region || "";
      data[i][outIdxMap.region_confidence] = res.region_confidence ?? "";
      data[i][outIdxMap.region_status]     = res.region_status || "";
      data[i][outIdxMap.rules_applied]     = res.rules_applied || "";

      const isBtLow = !res.bt_status || String(res.bt_status).toLowerCase().includes("low");
      const isGkLow = !res.gk_status || String(res.gk_status).toLowerCase().includes("low");
      const isRegionLow = !res.region_status || String(res.region_status).toLowerCase().includes("low");

      const btHl = isBtLow ? "#FFF2CC" : null;
      const gkHl = isGkLow ? "#FFF2CC" : null;
      const regionHl = isRegionLow ? "#FFF2CC" : null;
      const rulesHl = (isBtLow || isGkLow || isRegionLow) ? "#FFF2CC" : null;

      backgrounds[i][outIdxMap.suggested_bt]  = btHl;
      backgrounds[i][outIdxMap.bt_confidence] = btHl;
      backgrounds[i][outIdxMap.bt_status]     = btHl;

      backgrounds[i][outIdxMap.suggested_gk]  = gkHl;
      backgrounds[i][outIdxMap.gk_confidence] = gkHl;
      backgrounds[i][outIdxMap.gk_status]     = gkHl;

      backgrounds[i][outIdxMap.suggested_region]  = regionHl;
      backgrounds[i][outIdxMap.region_confidence] = regionHl;
      backgrounds[i][outIdxMap.region_status]     = regionHl;

      backgrounds[i][outIdxMap.rules_applied]     = rulesHl;

    } else if (task === "pipeline") {
      data[i][outIdxMap.suggested_gk]      = res.suggested_gk || "";
      data[i][outIdxMap.suggested_bt]      = res.suggested_bt || "";
      data[i][outIdxMap.suggested_region]  = res.suggested_region || "";
      data[i][outIdxMap.pipeline_source]   = res.pipeline_source || "";

      // Highlight escalated rows yellow; high-conf rows are uncoloured
      const hl = res.escalated ? "#FFF2CC" : null;
      backgrounds[i][outIdxMap.suggested_gk]      = hl;
      backgrounds[i][outIdxMap.suggested_bt]      = hl;
      backgrounds[i][outIdxMap.suggested_region]  = hl;
      if (res.escalated) {
        backgrounds[i][outIdxMap.pipeline_source] = "#D0E8FF"; // light blue = classifier was invoked
      } else {
        backgrounds[i][outIdxMap.pipeline_source] = null;
      }
    }
  }

  dataRange.setValues(data);
  dataRange.setBackgrounds(backgrounds);
}

// ─── Help ─────────────────────────────────────────────────────────────────────

function showInstructions() {
  const html = `
    <div style="font-family:sans-serif;font-size:13px;line-height:1.7;">
      <h3 style="color:#1e293b;">SKU MatchOps – Setup</h3>
      <ol>
        <li>Start the API server:<br>
          <code>python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload</code>
        </li>
        <li>Expose with Cloudflare Tunnel:<br>
          <code>cloudflared tunnel --url http://localhost:8000</code>
        </li>
        <li>Paste the Cloudflare Tunnel URL into <b>CONFIG.BASE_URL</b>.</li>
        <li>Deploy this script as a <b>Web App</b>:<br>
          Deploy → New Deployment → Web App<br>
          Execute as: <b>Me</b> &nbsp;|&nbsp; Who has access: <b>Anyone</b><br>
          Copy the URL into <b>CONFIG.CALLBACK_URL</b>.
        </li>
        <li>Run any mode — results appear automatically when done.</li>
      </ol>
      <p style="color:#64748b;"><b>All modes are fully async</b> (no 6-min timeout).<br>
      <b>Pipeline</b> = Matcher first, then Classifier for medium/low confidence rows.</p>
    </div>
  `;

  SpreadsheetApp.getUi().showModalDialog(
    HtmlService.createHtmlOutput(html).setWidth(500).setHeight(360),
    "Setup Instructions"
  );
}