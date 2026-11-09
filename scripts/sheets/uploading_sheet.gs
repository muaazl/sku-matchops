/**
 * SKU MatchOps – UPLOADING SCRIPT (Zero-Tunnel Queue Mode)
 * ==========================================
 * Paste this into the Apps Script editor of your UPLOADING Google Sheet.
 */

const CONFIG = {
  // Google Sheet ID of the central catalog sheet where the _queue tab lives.
  CATALOG_SHEET_ID: "17wPPEUu0_KbgOPtjDy4UVS8fh76D1ZmQlB5FZMw2JEo"
};

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("MatchOps Upload")
    .addItem("Clean & Upload SKUs", "runUpload")
    .addToUi();
}

/**
 * Returns the _queue sheet from the central catalog spreadsheet,
 * ensuring the 8-column schema (including "Type") is present.
 */
function _getOrCreateQueueSheet() {
  const ss = SpreadsheetApp.openById(CONFIG.CATALOG_SHEET_ID);
  let sheet = ss.getSheetByName("_queue");
  if (!sheet) {
    sheet = ss.insertSheet("_queue");
    sheet.getRange("A1:H1").setValues([["Job ID", "Spreadsheet ID", "Sheet Name", "Type", "Domain", "Task", "Status", "Created At"]]);
    sheet.getRange("A1:H1").setFontWeight("bold");
    sheet.setColumnWidth(1, 280);
    sheet.setColumnWidth(2, 280);
    sheet.setColumnWidth(3, 180);
    sheet.setColumnWidth(4, 110);
    sheet.setColumnWidth(5, 100);
    sheet.setColumnWidth(6, 100);
    sheet.setColumnWidth(7, 140);
    sheet.setColumnWidth(8, 180);
    return sheet;
  }

  const lastCol = Math.max(1, sheet.getLastColumn());
  const headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map(h => String(h).trim());
  if (!headers.some(h => h.toLowerCase() === "type")) {
    const sheetNameIdx = headers.findIndex(h => h.toLowerCase() === "sheet name");
    const insertAfter = sheetNameIdx !== -1 ? sheetNameIdx + 1 : 3;
    sheet.insertColumnAfter(insertAfter);
    sheet.getRange(1, insertAfter + 1).setValue("Type").setFontWeight("bold");
    sheet.setColumnWidth(insertAfter + 1, 110);
  }
  return sheet;
}

/**
 * Strips spaces after commas in the "Generic keywords" column (defaults to Column I / 9).
 */
function removeSpacesAfterCommas(sheet) {
  const lastRow = sheet.getLastRow();
  const lastCol = sheet.getLastColumn();
  if (lastRow < 2 || lastCol < 1) return;

  const headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map(h => String(h).trim().toLowerCase());
  const gkIdx = headers.indexOf("generic keywords");
  const colToClean = gkIdx !== -1 ? gkIdx + 1 : 9;
  if (colToClean > lastCol) return;

  const range = sheet.getRange(2, colToClean, lastRow - 1, 1);
  const values = range.getValues();

  for (let i = 0; i < values.length; i++) {
    if (typeof values[i][0] === "string") {
      values[i][0] = values[i][0].replace(/,\s+/g, ",");
    }
  }

  range.setValues(values);
}

/**
 * Cleans data and queues an upload job in the central _queue tab for local sync_sheets.py --watch.
 */
function runUpload() {
  const ui = SpreadsheetApp.getUi();
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getActiveSheet();
  const sheetName = sheet.getName();

  // 1. Extract Outlet ID from the sheet name (e.g., "91212_Coffee Luv...")
  const outletId = sheetName.split("_")[0].trim();

  if (!outletId || isNaN(parseInt(outletId, 10))) {
    ui.alert(
      "Error",
      "Cannot extract Outlet ID from the sheet name. Please ensure the tab name starts with the ID followed by an underscore (e.g., '91212_Name').",
      ui.ButtonSet.OK
    );
    return;
  }

  // 2. Clean spaces after commas in "Generic keywords"
  try {
    removeSpacesAfterCommas(sheet);
  } catch (e) {
    Logger.log("Failed to clean spaces: " + e.message);
  }

  // 3. Count valid SKUs
  const data = sheet.getDataRange().getValues();
  if (data.length < 2) {
    ui.alert("Notice", "No SKUs found to upload.", ui.ButtonSet.OK);
    return;
  }

  const headers = data[0].map(h => String(h).trim());
  const nameColIdx = Math.max(0, headers.findIndex(h => h.toLowerCase() === "name"));
  let skuCount = 0;
  for (let i = 1; i < data.length; i++) {
    if (String(data[i][nameColIdx] || "").trim() !== "") {
      skuCount++;
    }
  }

  if (skuCount === 0) {
    ui.alert("Notice", "No SKUs found to upload.", ui.ButtonSet.OK);
    return;
  }

  // Flush cleaned values before the local worker reads the sheet
  SpreadsheetApp.flush();

  // 4. Confirm with the user before queueing
  const confirmResponse = ui.alert(
    "Confirm Upload",
    `Are you sure you want to queue an upload for Outlet ID: ${outletId}?\n\nTotal SKUs to upload: ${skuCount}`,
    ui.ButtonSet.YES_NO
  );

  if (confirmResponse !== ui.Button.YES) {
    return;
  }

  // 5. Append job to the central _queue tab
  try {
    const jobId = Utilities.getUuid();
    const queueSheet = _getOrCreateQueueSheet();
    queueSheet.appendRow([
      jobId,
      ss.getId(),
      sheetName,
      "uploading",
      "food",
      "upload",
      "PENDING",
      new Date().toISOString()
    ]);

    ss.toast(
      `Upload queued for Outlet ${outletId} (${skuCount} SKUs). Local sync_sheets.py watcher will run Chrome automation.`,
      "⏳ MatchOps Upload",
      10
    );
  } catch (e) {
    ui.alert("Error", `Could not write to the central _queue tab.\n\nDetails: ${e.message}`, ui.ButtonSet.OK);
  }
}
