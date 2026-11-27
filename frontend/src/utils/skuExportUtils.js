import { formatConfidencePct, formatGkText } from './formatters';

export function sanitizeExportFilename(rawName = 'Sheet') {
  const baseName = rawName.replace(/\.[^/.]+$/, '');
  return baseName.replace(/[/\\?%*:|"<>]/g, '_').trim() || 'Sheet';
}

export function buildCleanExportRows(skusList, domain) {
  return skusList.map((row) => {
    const cleanGk = row._cleanGk ?? formatGkText(row.generic_keywords || row.gk_json || row.gk);
    if (domain === 'market') {
      return {
        'SKU Name': row.sku_name || '',
        Categories: row.categories || row.region || '',
        'Generic Keywords': cleanGk,
        'Basic Type': row.basic_type || row.bt || '',
      };
    }
    if (domain === 'food') {
      return {
        'SKU Name': row.sku_name || '',
        'Generic Keywords': cleanGk,
        'Basic Type': row.basic_type || row.bt || '',
        Region: row.region || '',
      };
    }
    return {
      'SKU Name': row.sku_name || '',
      Categories: row.categories || '',
      'Generic Keywords': cleanGk,
      'Basic Type': row.basic_type || row.bt || '',
      Region: row.region || '',
    };
  });
}

export function buildAuditExportRows(skusList) {
  return skusList.map((row) => ({
    ID: row.id,
    'Batch ID': row.batch_id,
    Domain: row.domain,
    'SKU Name': row.sku_name || '',
    'Generic Keywords': row._cleanGk ?? formatGkText(row.generic_keywords || row.gk_json || row.gk),
    'Basic Type': row.basic_type || row.bt || '',
    Region: row.region || '',
    Categories: row.categories || '',
    'Match Source': row.match_source || '',
    'Matched Catalog Name': row.matched_catalog_name || '',
    'Match Score': row.match_score != null ? row.match_score : '',
    'BT Confidence': row.bt_confidence != null ? row.bt_confidence : '',
    'GK Confidence': row.gk_confidence != null ? row.gk_confidence : '',
    'Region Confidence': row.region_confidence != null ? row.region_confidence : '',
    'Final Confidence': row.confidence != null ? row.confidence : '',
    'Logic Notes': row.logic_notes || '',
    'Rules Applied':
      typeof row.rules_applied_json === 'object'
        ? JSON.stringify(row.rules_applied_json)
        : row.rules_applied_json || '',
    'Created At': row.created_at || '',
  }));
}

export function buildSkuAuditMetadataRows(selectedSku, { isClassifier = false, isMatcher = false } = {}) {
  const srcLower = String(selectedSku.match_source || '')
    .trim()
    .toLowerCase();
  const isSkuMatcher = srcLower === 'matcher' || (!srcLower && isMatcher);
  const isPureClassifierJob = isClassifier && !selectedSku.matched_catalog_name && selectedSku.match_score == null;

  return [
    ['Source', selectedSku.match_source || '—'],
    ...(!isPureClassifierJob
      ? [
          ['Matched Catalog Name', selectedSku.matched_catalog_name || '—'],
          ['Logic Notes', selectedSku.logic_notes || '—'],
          ['Match Score', formatConfidencePct(selectedSku.match_score) || '—'],
        ]
      : []),
    ['BT Confidence', isSkuMatcher ? 'N/A' : formatConfidencePct(selectedSku.bt_confidence) || '—'],
    ['GK Confidence', isSkuMatcher ? 'N/A' : formatConfidencePct(selectedSku.gk_confidence) || '—'],
    ['Region/Cat Confidence', isSkuMatcher ? 'N/A' : formatConfidencePct(selectedSku.region_confidence) || '—'],
    ['Final Confidence', formatConfidencePct(selectedSku.confidence) || '—'],
  ].filter(([, v]) => v !== null && v !== undefined);
}
