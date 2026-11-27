import React from 'react';
import PropTypes from 'prop-types';
import { Button, Card, Collapse, Grid, InputAdornment, Stack, TextField } from '@mui/material';
import { Search, Tune } from '@mui/icons-material';
import { getCatalogDatasets } from '../../utils/catalogUtils';

export function CatalogFilterBar({
  domain,
  dataset,
  query,
  onQueryChange,
  showFilters,
  onToggleFilters,
  priceMin,
  onPriceMinChange,
  priceMax,
  onPriceMaxChange,
  region,
  onRegionChange,
  category,
  onCategoryChange,
  gkContains,
  onGkContainsChange,
  brandFilter,
  onBrandFilterChange,
  btFilter,
  onBtFilterChange,
  onClearAdvancedFilters,
}) {
  const datasets = getCatalogDatasets(domain);
  const currentDatasetLabel = datasets.find((d) => d.id === dataset)?.label || 'dataset';

  return (
    <Card sx={{ p: 2, mb: 3 }}>
      <Stack direction="row" spacing={2} sx={{ width: '100%' }}>
        <TextField
          size="small"
          placeholder={`Search ${currentDatasetLabel}...`}
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          sx={{ flexGrow: 1 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ fontSize: 16 }} />
              </InputAdornment>
            ),
          }}
        />

        {dataset === 'catalog' && (
          <Button
            variant="outlined"
            color={showFilters ? 'primary' : 'inherit'}
            startIcon={<Tune sx={{ fontSize: 16 }} />}
            onClick={onToggleFilters}
          >
            Filters
          </Button>
        )}
      </Stack>

      <Collapse in={showFilters && dataset === 'catalog'} sx={{ mt: 2 }}>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              label="Min Price"
              type="number"
              value={priceMin}
              onChange={(e) => onPriceMinChange(e.target.value)}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              label="Max Price"
              type="number"
              value={priceMax}
              onChange={(e) => onPriceMaxChange(e.target.value)}
            />
          </Grid>
          {domain === 'food' ? (
            <Grid size={{ xs: 12, sm: 4, md: 3 }}>
              <TextField
                size="small"
                fullWidth
                label="Region"
                value={region}
                onChange={(e) => onRegionChange(e.target.value)}
              />
            </Grid>
          ) : (
            <Grid size={{ xs: 12, sm: 4, md: 3 }}>
              <TextField
                size="small"
                fullWidth
                label="Categories"
                value={category}
                onChange={(e) => onCategoryChange(e.target.value)}
              />
            </Grid>
          )}
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              label="Generic Keywords"
              value={gkContains}
              onChange={(e) => onGkContainsChange(e.target.value)}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              label={domain === 'food' ? 'Flavor' : 'Brand'}
              value={brandFilter}
              onChange={(e) => onBrandFilterChange(e.target.value)}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              label="Basic Type"
              value={btFilter}
              onChange={(e) => onBtFilterChange(e.target.value)}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4, md: 3 }}>
            <Button fullWidth variant="text" color="secondary" onClick={onClearAdvancedFilters}>
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Collapse>
    </Card>
  );
}

CatalogFilterBar.propTypes = {
  domain: PropTypes.string.isRequired,
  dataset: PropTypes.string.isRequired,
  query: PropTypes.string.isRequired,
  onQueryChange: PropTypes.func.isRequired,
  showFilters: PropTypes.bool.isRequired,
  onToggleFilters: PropTypes.func.isRequired,
  priceMin: PropTypes.string.isRequired,
  onPriceMinChange: PropTypes.func.isRequired,
  priceMax: PropTypes.string.isRequired,
  onPriceMaxChange: PropTypes.func.isRequired,
  region: PropTypes.string.isRequired,
  onRegionChange: PropTypes.func.isRequired,
  category: PropTypes.string.isRequired,
  onCategoryChange: PropTypes.func.isRequired,
  gkContains: PropTypes.string.isRequired,
  onGkContainsChange: PropTypes.func.isRequired,
  brandFilter: PropTypes.string.isRequired,
  onBrandFilterChange: PropTypes.func.isRequired,
  btFilter: PropTypes.string.isRequired,
  onBtFilterChange: PropTypes.func.isRequired,
  onClearAdvancedFilters: PropTypes.func.isRequired,
};
