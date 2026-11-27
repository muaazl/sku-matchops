import { safeJsonParse } from './formatters';

function parseArrayField(value) {
  if (!value || value === '[]') return [];
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    const parsed = safeJsonParse(value, []);
    return Array.isArray(parsed) ? parsed : [];
  }
  return [];
}

export function extractAuditEntities(auditTrace) {
  const nerEntities = auditTrace?.stage1_nlp?.ner_entities;
  if (nerEntities) {
    return Object.entries(nerEntities).flatMap(([key, val]) =>
      Array.isArray(val) && val.length > 0
        ? val.map((item, idx) => ({ id: `${key}-${idx}`, label: `${key}: ${item}` }))
        : []
    );
  }

  const flavors = parseArrayField(auditTrace?.final_output?.flavor_extraction);
  const brands = parseArrayField(auditTrace?.final_output?.brand_extraction);

  return [
    ...flavors.map((item, idx) => ({ id: `flav-${idx}`, label: `flavor: ${item}` })),
    ...brands.map((item, idx) => ({ id: `brand-${idx}`, label: `brand: ${item}` })),
  ];
}

export function transformAuditToVectorResults(auditData) {
  return (auditData?.stage3_cross_encoder?.top_candidates || []).map((c, idx) => ({
    id: c.cand_idx ?? idx,
    score: c.raw_cross_score,
    raw_cross_score: c.raw_cross_score,
    payload: {
      Name: c.catalog_name,
      Description: c.description || '',
      SellerCategory: c.seller_category || c.region_category || '',
      basictype: c.basic_type,
      gk: c.generic_keywords,
      region: c.region_category,
      category: c.region_category,
      raw_cross_score: c.raw_cross_score,
      final_cross_score: c.final_cross_score,
    },
  }));
}
