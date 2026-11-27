export const TAG_EXAMPLES = [
  { sku: 'Egg Roll', gk: 'Short Eats, Rolls, Egg Roll', bt: 'Rolls', region: 'Sri Lankan' },
  { sku: 'Chicken Pizza', gk: 'Pizza, Chicken Pizza', bt: 'Pizza', region: 'Italian' },
  { sku: 'French Fries', gk: 'Fast Food, Fries', bt: 'Fries', region: 'Western' },
  { sku: 'Cappuccino', gk: 'Beverage, Coffee, Cappuccino', bt: 'Cappuccino', region: 'Sri Lankan' },
  { sku: 'Chicken Fried Rice', gk: 'Fried Rice, Chicken Fried Rice', bt: 'Fried Rice', region: 'Sri Lankan Chinese' },
  {
    sku: 'Sandwich Bread',
    gk: 'Sandwich Bread, White Sandwich Bread, Bakery',
    bt: 'Sandwich Bread',
    region: 'Sri Lankan',
  },
];

export const STATUS_ROWS = [
  {
    status: 'AUTO',
    range: '≥ 80%',
    color: 'success',
    meaning: 'Tag is trusted as-is and flows straight through the pipeline.',
  },
  {
    status: 'REVIEW',
    range: '50% – 79%',
    color: 'warning',
    meaning: 'Tag is kept but flagged — worth a second look before treating it as ground truth.',
  },
  {
    status: 'LOW',
    range: '< 50%',
    color: 'error',
    meaning: 'Tag is unreliable; the SKU is escalated rather than auto-approved.',
  },
];

export const SOURCE_GLOSSARY = [
  {
    source: 'trained',
    meaning: 'Predicted by a classifier trained on labeled catalog data — the default, highest-volume path.',
  },
  {
    source: 'zero-shot',
    meaning: 'Predicted by embedding similarity when no trained classifier confidently covers the SKU.',
  },
  { source: 'override', meaning: 'Forced by a Rules Engine rule — a human decided this SKU always gets this tag.' },
  { source: 'keyword', meaning: 'Matched directly off a keyword in the SKU name (fast, deterministic fallback).' },
  { source: 'synthetic', meaning: 'Injected automatically, e.g. combining brand + category into a Generic Keyword.' },
  {
    source: 'conflict',
    meaning: 'Two signals disagreed on the tag — surfaced so it can be resolved rather than silently picked.',
  },
];

export const OUTLET_EXAMPLES = [
  { outlet: 'Pizza Hut', region: 'Italian' },
  { outlet: 'Keells', region: 'Bakery' },
  { outlet: 'Chola Authentic Indian Restaurant', region: 'South Indian / North Indian' },
  { outlet: 'Hotel De Plaza', region: 'Sri Lankan Kottu' },
];
