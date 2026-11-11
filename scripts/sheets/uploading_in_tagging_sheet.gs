/**
 * Creates a custom menu in Google Sheets
 */
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('Muaaz Stuff')
    .addItem('Start Uploading Process', 'startUploadingProcess')
    .addItem('Validate Active Sheet', 'validateWords')
    .addItem('Include Keyword', 'addKeywordToSelection')
    .addSeparator()
    .addItem("Run MatchOps",      "runPipelineFood")
    .addToUi();
}

const EXTERNAL_DICTIONARY_SPREADSHEET_ID = "1P0Z8T3-5cjC8d3ZNKlcqKB5SckAApyMXMso1f8eoE6E";

const DICTIONARY_SHEET_NAMES = [
  "Food - GKs",
  "Food - BT Taggs",
  "Food - Cuisine Tags"
];

// Replaced fixed column numbers with column headers
const HEADERS_TO_VALIDATE = ['Generic keywords', 'basictype', 'region'];

const INVALID_CELL_COLOR = "#ffcdd2";

function validateWords() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let isValidRun = true; 
  
  const dictionaries = loadDictionaries();
  if (!dictionaries) return false; 

  const dataSheet = ss.getActiveSheet(); 
  if (!dataSheet) {
    SpreadsheetApp.getUi().alert('Error: No active sheet was found. Please ensure you have a sheet selected.');
    return false;
  }
  
  const dataRange = dataSheet.getDataRange();
  const cellValues = dataRange.getValues();
  
  if (cellValues.length === 0) return true; // Exit if the sheet is empty

  // Get the headers from the first row and find their 0-based column indices
  const headers = cellValues[0];
  const targetColIndices = HEADERS_TO_VALIDATE.map(header => headers.indexOf(header));
  
  // Create a matching 2D array to hold our new Rich Text values
  const richTextValues = [];

  for (let row = 0; row < cellValues.length; row++) {
    richTextValues[row] = [];
    
    for (let col = 0; col < cellValues[row].length; col++) {
      const currentColumnIndex = col + 1; 
      
      // Match the current column (0-based) to see if it's one of our target columns
      const dictIndex = targetColIndices.indexOf(col);
      const cellValue = cellValues[row][col];
      const cellString = String(cellValue);

      // 1. If it's the header, an unvalidated column, or empty, fill it and SKIP completely
      if (row === 0 || dictIndex === -1 || cellString.trim() === "") {
        richTextValues[row][col] = SpreadsheetApp.newRichTextValue()
          .setText(cellString)
          .build();
        continue; // This forcefully jumps to the next 'col' iteration
      }

      // 2. Only processes validation if dictIndex IS NOT -1
      const wordsInCell = cellString.split(',').map(word => word.trim());
      const builder = SpreadsheetApp.newRichTextValue().setText(cellString);
      
      const clearStyle = SpreadsheetApp.newTextStyle().setForegroundColor('#000000').setBold(false).build();
      if (cellString.length > 0) {
        builder.setTextStyle(0, cellString.length, clearStyle);
      }

      let cellHasInvalidWord = false;
      let searchStartIndex = 0; 

      wordsInCell.forEach(word => {
        if (word === "") return;

        const wordIndex = cellString.indexOf(word, searchStartIndex);
        
        if (wordIndex !== -1) {
          // Safe check
          if (dictionaries[dictIndex] && !dictionaries[dictIndex].has(word)) {
            const invalidStyle = SpreadsheetApp.newTextStyle()
              .setForegroundColor('#FF0000')
              .setBold(true)
              .build();
            
            const endIndex = wordIndex + word.length;
            builder.setTextStyle(wordIndex, endIndex, invalidStyle);
            cellHasInvalidWord = true;
            
            console.log(`Invalid word at row ${row + 1}, col ${currentColumnIndex} (${HEADERS_TO_VALIDATE[dictIndex]}):`, word);
          }
          searchStartIndex = wordIndex + word.length;
        }
      });

      if (cellHasInvalidWord) {
        isValidRun = false;
      }

      richTextValues[row][col] = builder.build();
    }
  }

  // Apply the rich text formatting to the entire range at once
  dataRange.setRichTextValues(richTextValues);
  return isValidRun; 
}

/**
 * Helper function to read words from dictionary sheets in the EXTERNAL file.
 */
function loadDictionaries() {
  const ui = SpreadsheetApp.getUi();

  let externalSs;
  try {
    externalSs = SpreadsheetApp.openById(EXTERNAL_DICTIONARY_SPREADSHEET_ID);
  } catch (e) {
    ui.alert(`Error: Could not open the external spreadsheet with ID: ${EXTERNAL_DICTIONARY_SPREADSHEET_ID}. Please check the ID and permissions. The error was: ${e.message}`);
    return null;
  }
  
  const dictionaries = [];

  for (const sheetName of DICTIONARY_SHEET_NAMES) {
    const dictSheet = externalSs.getSheetByName(sheetName);
    if (!dictSheet) {
      ui.alert(`Error: Dictionary sheet named "${sheetName}" was not found in the external file.`);
      return null;
    }
    
    const range = dictSheet.getRange("A1:A" + dictSheet.getLastRow());
    const values = range.getValues();
    const backgrounds = range.getBackgrounds();
    const defaultColor = "#ffffff"; 
    
    const validWords = [];
    for (let i = 0; i < values.length; i++) {
      const cellValue = values[i][0];
      const cellColor = backgrounds[i][0];
      
      if (cellValue && (cellColor === defaultColor || cellColor === "white")) {
        validWords.push(String(cellValue).trim());
      }
    }
    dictionaries.push(new Set(validWords));
  }
  return dictionaries;
}

/**
 * Main entry point triggered by the custom menu.
 */
function startUploadingProcess() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getActiveSheet();
  const tabName = sheet.getName();
  
  try {
    const isTaggingValid = validateWords();
    if (!isTaggingValid) {
      SpreadsheetApp.getUi().alert("Process Stopped", "Invalid tags detected. Please fix the highlighted words before fetching SKU data.", SpreadsheetApp.getUi().ButtonSet.OK);
      return; 
    }

    const token = getValidToken();
    searchMerchant(token, tabName, sheet.getSheetId());
  } catch (e) {
    console.error("[CRITICAL ERROR in startUploadingProcess]:", e.message, e.stack);
    SpreadsheetApp.getUi().alert("Process Stopped", e.message, SpreadsheetApp.getUi().ButtonSet.OK);
  }
}

/**
 * Handles the 3-hour token logic.
 */
function getValidToken() {
  const props = PropertiesService.getUserProperties();
  const tokenDataStr = props.getProperty('pickme_api_token');
  
  if (tokenDataStr) {
    const tokenData = JSON.parse(tokenDataStr);
    const now = new Date().getTime();
    if (now - tokenData.timestamp < 10800000) {
      return tokenData.token;
    }
  }
  
  const ui = SpreadsheetApp.getUi();
  const response = ui.prompt('API Token Expired/Missing', 'Please paste your raw Bearer token (without the word "Bearer"):', ui.ButtonSet.OK_CANCEL);
  
  if (response.getSelectedButton() == ui.Button.OK) {
    let token = response.getResponseText().trim();
    if (token === "") throw new Error("Token cannot be empty.");
    
    props.setProperty('pickme_api_token', JSON.stringify({
      token: token,
      timestamp: new Date().getTime()
    }));
    return token;
  } else {
    throw new Error("Token input cancelled.");
  }
}

/**
 * Searches for the merchant based on tab name.
 */
function searchMerchant(token, tabName, sheetId) {
  console.log(`[DEBUG - searchMerchant] Starting search for tabName: "${tabName}"`);
  
  const filterObj = [{"field":"search_keyword","value":tabName.trim(),"operator":1}];
  const filterStr = encodeURIComponent(JSON.stringify(filterObj));
  const pagingObj = {"page":1,"perPage":500};
  const pagingStr = encodeURIComponent(JSON.stringify(pagingObj));
  
  const url = `https://uni-portal-api.pickme.lk/food/v1/t0001/food/place/view?filters=${filterStr}&page=1&paging=${pagingStr}`;
  // const url = `https://food-portal-api-go.pickme.lk/v1/food/place/view?filters=${filterStr}&page=1&paging=${pagingStr}`;
  
  const options = {
    "method": "GET",
    "headers": {
      "Authorization": "Bearer " + token,
      "user-action": "view_restaurant/view"
    },
    "muteHttpExceptions": true
  };

  const response = UrlFetchApp.fetch(url, options);
  const responseCode = response.getResponseCode();
  const responseText = response.getContentText();
  
  let json;
  try {
    json = JSON.parse(responseText);
  } catch (e) {
    throw new Error(`Failed to parse Search Merchant API response (JSON error). Response code: ${responseCode}. Content: ${responseText.substring(0, 300)}`);
  }
  
  const lowerResponse = responseText.toLowerCase();
  const isTokenInvalid = responseCode === 401 || responseCode === 403 || 
    lowerResponse.includes("signature verification failed") ||
    (json && json.errors && json.errors.some(err => 
      err.code === "MER-4007" || err.code === "MER-4006" || err.code === "MER-4002" ||
      (err.message && (
        err.message.toLowerCase().includes("token blacklisted") || 
        err.message.toLowerCase().includes("token expired") ||
        err.message.toLowerCase().includes("signature")
      ))
    ));

  if (isTokenInvalid) {
    PropertiesService.getUserProperties().deleteProperty('pickme_api_token');
    throw new Error("Your API token has expired or has an invalid signature. The saved token has been cleared. Please restart the process to enter a fresh token.");
  }
  
  if (!json.data || json.data.length === 0) {
    const ui = SpreadsheetApp.getUi();
    const promptResponse = ui.prompt('Merchant Not Found', `No merchant found matching "${tabName}".\n\nPlease enter the Merchant ID manually to continue:`, ui.ButtonSet.OK_CANCEL);
    
    if (promptResponse.getSelectedButton() == ui.Button.OK) {
      const manualId = promptResponse.getResponseText().trim();
      if (manualId === "") {
        ui.alert("Process Cancelled", "Merchant ID cannot be empty.", ui.ButtonSet.OK);
        return;
      }
      processMerchantData(manualId, tabName, sheetId);
    }
    return;
  }
  
  if (json.data.length === 1) {
    processMerchantData(json.data[0].id, tabName, sheetId);
  } else {
    showMerchantPicker(json.data, tabName, sheetId);
  }
}

/**
 * Displays the HTML modal to choose a merchant.
 */
function showMerchantPicker(merchants, tabName, sheetId) {
  const template = HtmlService.createTemplateFromFile('Picker');
  template.merchants = merchants;
  template.tabName = tabName;
  template.sheetId = sheetId;
  
  const html = template.evaluate()
      .setWidth(500)
      .setHeight(400);
      
  SpreadsheetApp.getUi().showModalDialog(html, 'Multiple Merchants Found');
}

/**
 * Core function to fetch CSV, merge old data, highlight, and create upload sheet.
 */
function processMerchantData(merchantId, tabName, sheetId) {
  const TARGET_SPREADSHEET_ID = "1VLZ4IKk6ciaxy0n_8xyX-pAT7GoOQaXUWk5CCDDIHgs";
  
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const oldSheet = getSheetById(ss, sheetId);
  
  if (!oldSheet) {
    SpreadsheetApp.getUi().alert("Error: Original sheet not found.");
    return;
  }
  
  const token = getValidToken(); 
  const csvUrl = `https://uni-portal-api.pickme.lk/food/v1/t0001/food/place/skus/csv/${merchantId}`
  // const csvUrl = `https://food-portal-api-go.pickme.lk/v1/food/place/skus/csv/${merchantId}`;
  
  const options = {
    "method": "GET",
    "headers": { 
      "Authorization": "Bearer " + token,
      "user-action": "view_restaurant/view"
    },
    "muteHttpExceptions": true
  };
  
  const response = UrlFetchApp.fetch(csvUrl, options);
  const responseCode = response.getResponseCode();
  const responseText = response.getContentText();
  
  const lowerResponse = responseText.toLowerCase();
  if (responseCode === 401 || responseCode === 403 || lowerResponse.includes("signature verification failed")) {
    PropertiesService.getUserProperties().deleteProperty('pickme_api_token');
    throw new Error("Your API token has expired or has an invalid signature. The saved token has been cleared. Please restart the process to enter a fresh token.");
  }

  if (responseText.trim().startsWith('{')) {
    try {
      const json = JSON.parse(responseText);
      if (json && json.errors) {
        const isBlacklisted = json.errors.some(err => 
          err.code === "MER-4007" || 
          (err.message && err.message.toLowerCase().includes("token blacklisted"))
        );
        if (isBlacklisted) {
          PropertiesService.getUserProperties().deleteProperty('pickme_api_token');
          throw new Error("Your API token has been blacklisted. The bad token has been cleared. Please restart the uploading process to enter a fresh token.");
        }
      }
    } catch (e) {}
  }
  
  let csvData;
  try {
    csvData = Utilities.parseCsv(responseText);
  } catch (e) {
    throw new Error(`Failed to parse CSV data for Merchant ID: ${merchantId}. Parse error: ${e.message}. Raw text sample: ${responseText.substring(0, 300)}`);
  }
  
  if (!csvData || csvData.length < 2) {
    SpreadsheetApp.getUi().alert(`No SKU data returned or CSV is empty.\n\nAPI Content: ${responseText.substring(0, 200)}`);
    return;
  }
  
  const sellableIdx = csvData[0].indexOf('Individually Sellable');
  if (sellableIdx > -1) {
    csvData = csvData.filter((row, index) => index === 0 || String(row[sellableIdx]).trim().toLowerCase() !== 'no');
  }
  
  const oldValues = oldSheet.getDataRange().getValues();
  const oldHeaders = oldValues[0];
  const oldIdx = {
    name: oldHeaders.indexOf('Name'),
    genKw: oldHeaders.indexOf('Generic keywords'),
    basicType: oldHeaders.indexOf('basictype'),
    region: oldHeaders.indexOf('region')
  };
  
  const oldRecords = {};
  for (let i = 1; i < oldValues.length; i++) {
    let name = oldValues[i][oldIdx.name];
    if (name) {
      oldRecords[name] = {
        genKw: oldIdx.genKw > -1 ? oldValues[i][oldIdx.genKw] : "",
        basicType: oldIdx.basicType > -1 ? oldValues[i][oldIdx.basicType] : "",
        region: oldIdx.region > -1 ? oldValues[i][oldIdx.region] : ""
      };
    }
  }
  
  const newHeaders = csvData[0];
  const newIdx = {
    name: newHeaders.indexOf('Name'),
    genKw: newHeaders.indexOf('Generic keywords'),
    basicType: newHeaders.indexOf('basictype'),
    region: newHeaders.indexOf('region'),
    refId: newHeaders.indexOf('RefID')
  };
  
  if (newIdx.name === -1 || newIdx.refId === -1) {
    throw new Error(`Critical column headers ('Name' or 'RefID') were missing from the API response. Found columns: ${newHeaders.join(', ')}`);
  }
  
  const colors = csvData.map(row => Array(row.length).fill(null));
  let stats = { newSku: 0, missingRef: 0, dupRef: 0 };
  let seenRefs = {};
  
  for (let i = 1; i < csvData.length; i++) {
    let row = csvData[i];
    let name = row[newIdx.name];
    let refId = row[newIdx.refId];
    
    if (oldRecords[name]) {
      if (newIdx.genKw > -1) row[newIdx.genKw] = oldRecords[name].genKw;
      if (newIdx.basicType > -1) row[newIdx.basicType] = oldRecords[name].basicType;
      if (newIdx.region > -1) row[newIdx.region] = oldRecords[name].region;
    } else {
      stats.newSku++;
      colors[i].fill('#d9ead3'); 
    }
    
    let refIdVal = (refId || "").toString().trim();
    if (refIdVal === "") {
      stats.missingRef++;
      colors[i][newIdx.refId] = '#f4cccc'; 
    } else {
      if (!seenRefs[refIdVal]) {
        seenRefs[refIdVal] = [i];
      } else {
        seenRefs[refIdVal].push(i);
      }
    }
  }
  
  for (let key in seenRefs) {
    if (seenRefs[key].length > 1) {
      stats.dupRef += seenRefs[key].length;
      seenRefs[key].forEach(rowIndex => {
        colors[rowIndex][newIdx.refId] = '#fff2cc'; 
      });
    }
  }
  
  let targetSs;
  try {
    targetSs = SpreadsheetApp.openById(TARGET_SPREADSHEET_ID);
  } catch (e) {
    SpreadsheetApp.getUi().alert("Error: Could not access the target Uploading sheet. Check the ID and permissions.");
    return;
  }

  const newSheetName = merchantId + "_" + tabName;
  let uploadSheet;
  
  try {
    uploadSheet = targetSs.insertSheet(newSheetName, targetSs.getSheets().length);
  } catch (e) {
    SpreadsheetApp.getUi().alert("Space limit reached or sheet creation failed on the uploading document. Please create a new Spreadsheet for future tagging.");
    return;
  }
  
  const targetRange = uploadSheet.getRange(1, 1, csvData.length, csvData[0].length);
  targetRange.setNumberFormat("@");
  targetRange.setValues(csvData);
  targetRange.setBackgrounds(colors);
  
  uploadSheet.getRange(1, 1, 1, csvData[0].length).setFontWeight("bold");
  uploadSheet.autoResizeColumns(1, csvData[0].length);
  
  // 1. Define the merchant portal URL
  const targetUrl = `https://uni-portal.pickme.lk/#/t0001/food-portal/food-manage/restaurants/${merchantId}/manage-skus`;
  // const targetUrl = `https://food-portal.pickme.lk/#/sku/${merchantId}/list`;
  
  // 2. Create an HTML modal that shows stats, tries to auto-open the link, and provides a fallback link
  const htmlContent = `
    <div style="font-family: Arial, sans-serif; padding: 10px;">
      <h3 style="margin-top: 0; color: #2e7d32;">Process Complete!</h3>
      <p>Data pushed to Uploading Sheet.</p>
      <ul>
        <li><strong>New SKUs:</strong> ${stats.newSku}</li>
        <li><strong>Missing RefIDs:</strong> ${stats.missingRef}</li>
        <li><strong>Duplicate RefIDs:</strong> ${stats.dupRef}</li>
      </ul>
      <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
      <p style="color: #555; font-size: 13px;">Opening Merchant Portal in a new tab...</p>
      <p style="font-size: 14px; background: #f1f8e9; padding: 10px; border-radius: 4px;">
        ⚠️ <em>If the new tab didn't open automatically (due to a pop-up blocker), 
        <a href="${targetUrl}" target="_blank" style="color: #1a73e8; font-weight: bold;">click here to open it manually</a>.</em>
      </p>
      <div style="text-align: right; margin-top: 20px;">
        <button onclick="google.script.host.close()" style="padding: 8px 16px; cursor: pointer; background: #eee; border: 1px solid #ccc; border-radius: 4px;">Close</button>
      </div>
    </div>
    <script>
      // Attempt to auto-open the new tab
      window.open('${targetUrl}', '_blank');
    </script>
  `;

  const htmlOutput = HtmlService.createHtmlOutput(htmlContent)
    .setWidth(450)
    .setHeight(350);

  SpreadsheetApp.getUi().showModalDialog(htmlOutput, 'Success: Upload Complete');
}

function getSheetById(ss, id) {
  const sheets = ss.getSheets();
  for (let i = 0; i < sheets.length; i++) {
    if (sheets[i].getSheetId() == id) return sheets[i];
  }
  return null;
}

function addKeywordToSelection() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var range = sheet.getActiveRange();
  
  if (!range) {
    SpreadsheetApp.getUi().alert('Please select some cells first.');
    return;
  }
  
  var ui = SpreadsheetApp.getUi();
  var response = ui.prompt(
    'Add Keyword', 
    'Enter the keyword you want to add to the selected cells:', 
    ui.ButtonSet.OK_CANCEL
  );
  
  if (response.getSelectedButton() == ui.Button.OK) {
    var keyword = response.getResponseText().trim();
    if (keyword === "") {
      ui.alert('Keyword cannot be empty.');
      return;
    }
    
    var values = range.getValues();
    
    for (var i = 0; i < values.length; i++) {
      for (var j = 0; j < values[i].length; j++) {
        var cellValue = values[i][j].toString().trim();
        if (cellValue === "") {
          values[i][j] = keyword;
        } else {
          values[i][j] = cellValue + ", " + keyword;
        }
      }
    }
    range.setValues(values);
  }
}

function resetTokenNow() {
  PropertiesService.getUserProperties().deleteProperty('pickme_api_token');
  SpreadsheetApp.getUi().alert("Success", "Bad token deleted! Now run your uploading process.", SpreadsheetApp.getUi().ButtonSet.OK);
}