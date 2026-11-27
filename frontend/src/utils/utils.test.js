import { describe, expect, it } from 'vitest';
import { extractAuditEntities, transformAuditToVectorResults } from './auditUtils';
import { mapRows, parseCSV, toCsv } from './csvUtils';
import {
  fmtDuration,
  fmtEta,
  formatGk,
  normalizeConfidencePct,
  normalizeMatchSources,
  parseUtcTimestamp,
} from './formatters';
import { buildConditionGroups, generateSampleForRule, normalizeConditions } from './ruleUtils';

describe('csvUtils', () => {
  it('parses quoted CSV rows and maps aliases', () => {
    const csv = 'title,cost,desc,cat\n"Chicken, Spicy",450,"Hot dish",Food';
    const { headers, rows } = parseCSV(csv);
    expect(headers).toEqual(['title', 'cost', 'desc', 'cat']);
    expect(mapRows(headers, rows)).toEqual([
      { name: 'Chicken, Spicy', price: '450', description: 'Hot dish', category: 'Food' },
    ]);
  });

  it('escapes commas, quotes, and newlines when serializing CSV', () => {
    const serialized = toCsv([{ name: 'Item "A", 1', price: '10' }]);
    expect(serialized).toBe('name,price\n"Item ""A"", 1",10');
  });
});

describe('formatters', () => {
  it('formats durations, ETAs, UTC timestamps, and generic keywords', () => {
    expect(fmtEta(65)).toBe('1m 5s');
    expect(fmtEta(0)).toBe('done');
    expect(fmtDuration(1.5)).toBe('1m 30s');
    expect(parseUtcTimestamp('2026-01-01 10:00:00')?.toISOString()).toBe('2026-01-01T10:00:00.000Z');
    expect(formatGk('["Dairy", "Butter"]')).toBe('Dairy, Butter');
    expect(normalizeConfidencePct(0.854)).toBe(85);
    expect(normalizeConfidencePct(92)).toBe(92);
  });

  it('aggregates raw match source counts into Classifier vs Matcher percentages', () => {
    expect(
      normalizeMatchSources([
        { source: 'classifier', count: 3 },
        { source: 'matcher', count: 5 },
        { source: 'hybrid', count: 2 },
      ])
    ).toEqual([
      { source: 'Classifier', count: 3, pct: 30 },
      { source: 'Matcher', count: 7, pct: 70 },
    ]);
  });
});

describe('ruleUtils', () => {
  it('normalizes AND/OR groups and generates matching sample SKU records', () => {
    const normalized = normalizeConditions(
      [
        { condition_type: 'sku_contains', value: 'prawn', negate: 0, operator: 'AND' },
        { condition_type: 'sku_contains', value: 'bun', negate: 1, operator: 'OR' },
      ],
      'AND',
      false
    );
    expect(normalized[0].condition_group).toBe(1);
    expect(normalized[1].condition_group).toBe(2);

    const groups = buildConditionGroups(normalized);
    expect(groups).toHaveLength(2);

    const sample = generateSampleForRule({
      domain: 'food',
      conditions: [normalized[0]],
    });
    expect(sample.sku_name).toContain('prawn');
    expect(sample.domain).toBe('food');
  });
});

describe('auditUtils', () => {
  it('extracts NER entities and transforms audit candidates into vector results', () => {
    expect(
      extractAuditEntities({
        stage1_nlp: { ner_entities: { brand: ['Anchor'], flavor: [] } },
      })
    ).toEqual([{ id: 'brand-0', label: 'brand: Anchor' }]);

    expect(
      extractAuditEntities({
        final_output: { flavor_extraction: '["Chocolate"]', brand_extraction: ['Milo'] },
      })
    ).toEqual([
      { id: 'flav-0', label: 'flavor: Chocolate' },
      { id: 'brand-0', label: 'brand: Milo' },
    ]);

    const results = transformAuditToVectorResults({
      stage3_cross_encoder: {
        top_candidates: [{ cand_idx: 7, catalog_name: 'Anchor Butter', raw_cross_score: 6.2 }],
      },
    });
    expect(results[0].id).toBe(7);
    expect(results[0].payload.Name).toBe('Anchor Butter');
  });
});
