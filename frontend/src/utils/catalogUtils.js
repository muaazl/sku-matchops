export function getCatalogDatasets(domain) {
  return [
    { id: 'catalog', label: 'Catalog SKUs' },
    { id: 'gk', label: 'Generic Keywords' },
    { id: 'bt', label: 'Basic Type' },
    { id: 'category', label: domain === 'food' ? 'Region' : 'Categories' },
    { id: 'brands', label: domain === 'food' ? 'Flavors List' : 'Brands List' },
    { id: 'bt_gk_map', label: 'Basic Type - Generic Keywords Map' },
  ];
}

export function getCatalogMetricCards(domain) {
  return [
    { id: 'catalog', label: 'Catalog SKUs', statKey: 'catalog' },
    { id: 'gk', label: 'Generic Keywords', statKey: 'gk' },
    { id: 'bt', label: 'Basic Type', statKey: 'bt' },
    { id: 'category', label: domain === 'food' ? 'Region' : 'Categories', statKey: 'category' },
    { id: 'brands', label: domain === 'food' ? 'Flavors List' : 'Brands List', statKey: 'brands' },
    { id: 'bt_gk_map', label: 'Basic Type - Generic Keywords Map', statKey: 'bt_gk_map' },
  ];
}

const formatPrice = (value) => {
  if (value == null) return '';
  const num = Number(value);
  return Number.isNaN(num) ? String(value) : num.toFixed(2);
};

const formatCount = (value) => (value != null ? value.toLocaleString() : '0');

const formatYesNo = ({ value }) => (value ? 'Yes' : 'No');

export function getCatalogColumns(dataset, domain) {
  switch (dataset) {
    case 'catalog':
      if (domain === 'market') {
        return [
          { field: 'name', headerName: 'SKU Name', flex: 1.5, minWidth: 260 },
          { field: 'category', headerName: 'Categories', width: 140 },
          { field: 'generic_keywords', headerName: 'Generic Keywords', flex: 1.2, minWidth: 250 },
          { field: 'basictype', headerName: 'Basic Type', width: 150 },
          { field: 'brand', headerName: 'Brand', width: 140 },
          { field: 'price', headerName: 'Price', width: 100, valueFormatter: formatPrice },
          { field: 'merchant', headerName: 'Merchant', width: 130 },
        ];
      }
      return [
        { field: 'name', headerName: 'SKU Name', flex: 1.5, minWidth: 260 },
        { field: 'generic_keywords', headerName: 'Generic Keywords', flex: 1.2, minWidth: 250 },
        { field: 'basictype', headerName: 'Basic Type', width: 150 },
        { field: 'region', headerName: 'Region', width: 120 },
        { field: 'flavor', headerName: 'Flavor', width: 140 },
        { field: 'price', headerName: 'Price', width: 100, valueFormatter: formatPrice },
        { field: 'description', headerName: 'Description', flex: 1, minWidth: 200 },
      ];
    case 'gk':
      return [
        { field: 'name', headerName: 'Generic Keywords Tag', flex: 1, minWidth: 250 },
        {
          field: 'count',
          headerName: 'Occurrence SKU Count',
          width: 180,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
      ];
    case 'bt':
      return [
        { field: 'name', headerName: 'Basic Type Tag', flex: 1, minWidth: 250 },
        {
          field: 'count',
          headerName: 'Occurrence SKU Count',
          width: 180,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
      ];
    case 'category':
      return [
        { field: 'name', headerName: domain === 'food' ? 'Region' : 'Categories', flex: 1, minWidth: 250 },
        {
          field: 'count',
          headerName: 'Occurrence SKU Count',
          width: 180,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
      ];
    case 'brands':
      if (domain === 'market') {
        return [
          { field: 'name', headerName: 'Brand Name', flex: 1, minWidth: 180 },
          { field: 'aliases', headerName: 'Aliases', flex: 1, minWidth: 200 },
          { field: 'is_weak', headerName: 'Weak Brand?', width: 130, renderCell: formatYesNo },
          {
            field: 'count',
            headerName: 'Occurrence SKU Count',
            width: 180,
            align: 'right',
            headerAlign: 'right',
            valueFormatter: formatCount,
          },
        ];
      }
      return [
        { field: 'name', headerName: 'Flavor Name', flex: 1, minWidth: 150 },
        { field: 'aliases', headerName: 'Aliases', flex: 1, minWidth: 200 },
        { field: 'is_meat', headerName: 'Is Meat?', width: 110, renderCell: formatYesNo },
        { field: 'is_vegetable', headerName: 'Is Vegetable?', width: 135, renderCell: formatYesNo },
        { field: 'is_seafood', headerName: 'Is Seafood?', width: 130, renderCell: formatYesNo },
        {
          field: 'count',
          headerName: 'Occurrence SKU Count',
          width: 180,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
      ];
    case 'bt_gk_map':
      return [
        { field: 'bt', headerName: 'Basic Type', flex: 1, minWidth: 180 },
        { field: 'gks', headerName: 'Mapped Generic Keywords', flex: 2, minWidth: 350 },
        {
          field: 'gk_count',
          headerName: 'Mapped Generic Keywords Count',
          width: 160,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
        {
          field: 'count',
          headerName: 'Associated SKU Count',
          width: 190,
          align: 'right',
          headerAlign: 'right',
          valueFormatter: formatCount,
        },
      ];
    default:
      return [];
  }
}

export function resolveTagSuggestionTarget(type, ruleDomain) {
  if (type === 'bt_is' || type === 'set_bt') {
    return { dataset: 'bt', domains: ruleDomain ? [ruleDomain] : [] };
  }
  if (type === 'gk_contains' || type === 'add_gk' || type === 'remove_gk') {
    return { dataset: 'gk', domains: ruleDomain ? [ruleDomain] : [] };
  }
  if (type === 'region_is' || type === 'set_region') {
    return { dataset: 'category', domains: ['food'] };
  }
  if (type === 'category_contains' || type === 'set_category') {
    return { dataset: 'category', domains: ['market'] };
  }
  if (type === 'flavor_contains') {
    return { dataset: 'brands', domains: ['food'] };
  }
  return { dataset: null, domains: [] };
}
