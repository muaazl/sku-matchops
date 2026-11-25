// Stable fallback avoids re-triggering state sync loops while rules query is loading.
export const EMPTY_RULES = [];

export function normalizeConditions(conditions = [], fallbackLogic = 'AND', fromServer = false) {
  let group = 1;
  const fallback = String(fallbackLogic || 'AND').toUpperCase();
  return conditions.map((c, idx) => {
    let rawOp = c.operator || (idx > 0 ? fallback : 'AND') || 'AND';
    if (
      fromServer &&
      idx > 0 &&
      rawOp.toUpperCase() === 'AND' &&
      fallback === 'OR' &&
      c.condition_group !== conditions[idx - 1]?.condition_group
    ) {
      rawOp = 'OR';
    }
    const operator = idx === 0 ? 'AND' : rawOp.toUpperCase() === 'OR' ? 'OR' : 'AND';
    if (idx > 0 && operator === 'OR') {
      group += 1;
    }
    return {
      ...c,
      operator,
      condition_group: group,
      negate: c.negate ? 1 : 0,
    };
  });
}

export function buildConditionGroups(conditions = []) {
  const groups = [];
  conditions.forEach((c, index) => {
    const op = index === 0 ? 'AND' : (c.operator || 'AND').toUpperCase();
    if (index === 0 || op === 'OR') {
      groups.push([{ cond: c, index }]);
    } else {
      groups[groups.length - 1].push({ cond: c, index });
    }
  });
  return groups;
}

export function getConditionPlaceholder(conditionType) {
  if (conditionType === 'price_below' || conditionType === 'price_above') return 'e.g. 250';
  if (conditionType === 'flavor_is') return 'Select flavor type...';
  return 'Type or select value...';
}

export function createEmptyRule(currentRules = []) {
  const maxPriority = currentRules.length > 0 ? Math.max(...currentRules.map((r) => r.priority || 0)) : 0;
  return {
    rule_id: `new_rule_${Math.floor(Math.random() * 900 + 100)}`,
    domain: 'market',
    priority: maxPriority + 10,
    description: '',
    reasoning: '',
    condition_logic: 'AND',
    is_active: 1,
    conditions: [{ condition_group: 1, operator: 'AND', condition_type: 'sku_contains', value: '', negate: 0 }],
    actions: [{ action_type: 'set_bt', value: '' }],
  };
}

export function generateSampleForRule(rule) {
  if (!rule) return null;
  const sampleObj = {
    sku_name: 'Sample Product 250ml',
    domain: rule.domain || 'market',
    price: 10.0,
    category: 'General',
    gk: [],
    bt: '',
    region: '',
  };

  if (rule.conditions && rule.conditions.length > 0) {
    const skuKeywords = [];
    const excludedKeywords = [];
    const groups = buildConditionGroups(rule.conditions);
    const activeGroup = groups.find((grp) => grp.some(({ cond }) => (cond.value || '').trim() !== '')) || groups[0];

    activeGroup.forEach(({ cond: c }) => {
      const val = (c.value || '').trim();
      if (!val) return;

      if (c.condition_type === 'sku_contains') {
        if (!c.negate) {
          skuKeywords.push(val);
        } else {
          excludedKeywords.push(val.toLowerCase());
        }
      } else if (c.condition_type === 'bt_is') {
        sampleObj.bt = c.negate ? 'Other Type' : val;
      } else if (c.condition_type === 'gk_contains') {
        if (!c.negate) sampleObj.gk.push(val);
      } else if (c.condition_type === 'category_contains') {
        sampleObj.category = c.negate ? 'Other Category' : val;
      } else if (c.condition_type === 'region_is') {
        sampleObj.region = c.negate ? 'Other Region' : val;
      } else if (c.condition_type === 'price_below') {
        const p = parseFloat(val);
        if (!isNaN(p)) sampleObj.price = Math.max(0, p - 1);
      } else if (c.condition_type === 'price_above') {
        const p = parseFloat(val);
        if (!isNaN(p)) sampleObj.price = p + 5;
      } else if (c.condition_type === 'flavor_contains' || c.condition_type === 'flavor_is') {
        if (!c.negate) {
          const lower = val.toLowerCase();
          if (lower === 'seafood') {
            skuKeywords.push('Prawn Curry');
          } else if (lower === 'meat') {
            skuKeywords.push('Chicken');
          } else if (lower === 'vegetable' || lower === 'veg') {
            skuKeywords.push('Potato');
          } else {
            skuKeywords.push(val);
          }
        }
      }
    });

    const cleanKeywords = skuKeywords.filter((kw) => !excludedKeywords.some((ex) => kw.toLowerCase().includes(ex)));
    if (cleanKeywords.length > 0) {
      sampleObj.sku_name = `${cleanKeywords.join(' ')} 250ml`;
    }
  }

  return sampleObj;
}
