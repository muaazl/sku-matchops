import React, { useMemo, useState } from 'react';
import { Box, Button, CircularProgress, Grid, Stack, Tab, Tabs } from '@mui/material';
import { Refresh, Search } from '@mui/icons-material';
import { CatalogDataTable } from '../components/catalog/CatalogDataTable';
import { CatalogFilterBar } from '../components/catalog/CatalogFilterBar';
import { CatalogMetricsGrid } from '../components/catalog/CatalogMetricsGrid';
import { CatalogSyncDiffDialog } from '../components/catalog/CatalogSyncDiffDialog';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { useCatalogSearch } from '../hooks/useCatalogSearch';
import { getCatalogColumns } from '../utils/catalogUtils';

export default function CatalogSearch() {
  const [domain, setDomain] = useState('market');
  const [dataset, setDataset] = useState('catalog');
  const [query, setQuery] = useState('');

  const [showFilters, setShowFilters] = useState(false);
  const [priceMin, setPriceMin] = useState('');
  const [priceMax, setPriceMax] = useState('');
  const [region, setRegion] = useState('');
  const [category, setCategory] = useState('');
  const [gkContains, setGkContains] = useState('');
  const [brandFilter, setBrandFilter] = useState('');
  const [btFilter, setBtFilter] = useState('');

  const [paginationModel, setPaginationModel] = useState({ page: 0, pageSize: 25 });
  const [sortModel, setSortModel] = useState([]);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const activeFilters = useMemo(
    () => ({
      query,
      priceMin,
      priceMax,
      region,
      category,
      gkContains,
      brandFilter,
      btFilter,
    }),
    [query, priceMin, priceMax, region, category, gkContains, brandFilter, btFilter]
  );

  const {
    summaryStats,
    data,
    isLoading,
    isChecking,
    isSyncing,
    diffData,
    diffDialogOpen,
    setDiffDialogOpen,
    checkSync,
    syncCache,
  } = useCatalogSearch({
    dataset,
    domain,
    filters: activeFilters,
    page: paginationModel.page,
    pageSize: paginationModel.pageSize,
    sortModel,
  });

  const columns = useMemo(() => getCatalogColumns(dataset, domain), [dataset, domain]);

  const handleHeaderClick = (col) => {
    if (col.sortable === false) return;
    const currentSort = sortModel[0];
    if (!currentSort || currentSort.field !== col.field) {
      setSortModel([{ field: col.field, sort: 'asc' }]);
    } else if (currentSort.sort === 'asc') {
      setSortModel([{ field: col.field, sort: 'desc' }]);
    } else {
      setSortModel([]);
    }
  };

  const clearAdvancedFilters = () => {
    setPriceMin('');
    setPriceMax('');
    setRegion('');
    setCategory('');
    setGkContains('');
    setBrandFilter('');
    setBtFilter('');
    setPaginationModel((prev) => ({ ...prev, page: 0 }));
  };

  const clearAllFilters = () => {
    setQuery('');
    clearAdvancedFilters();
  };

  const hasActiveFilters = Boolean(
    query || priceMin || priceMax || region || category || gkContains || brandFilter || btFilter
  );

  return (
    <PageContainer>
      <PageHeader
        title="Catalog Search"
        subtitle="Explore, search, and audit reference catalogs, keywords, brands/flavors, and dynamic mappings."
        actions={
          <Stack direction="row" spacing={1.5}>
            <Button
              variant="outlined"
              color="primary"
              startIcon={isChecking ? <CircularProgress size={16} color="inherit" /> : <Search sx={{ fontSize: 16 }} />}
              disabled={isChecking || isSyncing}
              onClick={checkSync}
            >
              {isChecking ? 'Checking...' : 'Check for Updates'}
            </Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={isSyncing ? <CircularProgress size={16} color="inherit" /> : <Refresh sx={{ fontSize: 16 }} />}
              disabled={isSyncing}
              onClick={() => setConfirmOpen(true)}
            >
              {isSyncing ? 'Syncing...' : 'Sync from Sheets'}
            </Button>
          </Stack>
        }
      />

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs
          value={domain}
          onChange={(e, newDomain) => {
            if (newDomain) {
              setDomain(newDomain);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }
          }}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Market Domain" value="market" sx={{ fontWeight: 600, textTransform: 'capitalize' }} />
          <Tab label="Food Domain" value="food" sx={{ fontWeight: 600, textTransform: 'capitalize' }} />
        </Tabs>
      </Box>

      <CatalogMetricsGrid
        domain={domain}
        activeDataset={dataset}
        summaryStats={summaryStats}
        onSelectDataset={(newDataset) => {
          setDataset(newDataset);
          setPaginationModel((prev) => ({ ...prev, page: 0 }));
        }}
      />

      <Grid container spacing={3}>
        <Grid size={{ xs: 12 }}>
          <CatalogFilterBar
            domain={domain}
            dataset={dataset}
            query={query}
            onQueryChange={(val) => {
              setQuery(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            showFilters={showFilters}
            onToggleFilters={() => setShowFilters((prev) => !prev)}
            priceMin={priceMin}
            onPriceMinChange={(val) => {
              setPriceMin(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            priceMax={priceMax}
            onPriceMaxChange={(val) => {
              setPriceMax(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            region={region}
            onRegionChange={(val) => {
              setRegion(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            category={category}
            onCategoryChange={(val) => {
              setCategory(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            gkContains={gkContains}
            onGkContainsChange={(val) => {
              setGkContains(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            brandFilter={brandFilter}
            onBrandFilterChange={(val) => {
              setBrandFilter(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            btFilter={btFilter}
            onBtFilterChange={(val) => {
              setBtFilter(val);
              setPaginationModel((prev) => ({ ...prev, page: 0 }));
            }}
            onClearAdvancedFilters={clearAdvancedFilters}
          />

          <Box sx={{ mb: 3 }}>
            <CatalogDataTable
              columns={columns}
              rows={data?.results || []}
              total={data?.total || 0}
              isLoading={isLoading}
              hasActiveFilters={hasActiveFilters}
              sortModel={sortModel}
              onHeaderClick={handleHeaderClick}
              onClearAllFilters={clearAllFilters}
              page={paginationModel.page}
              pageSize={paginationModel.pageSize}
              onPageChange={(e, newPage) => setPaginationModel((prev) => ({ ...prev, page: newPage }))}
              onRowsPerPageChange={(e) =>
                setPaginationModel((prev) => ({
                  ...prev,
                  pageSize: parseInt(e.target.value, 10),
                  page: 0,
                }))
              }
            />
          </Box>
        </Grid>
      </Grid>

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title="Sync from Google Sheets"
        message={
          'Are you sure you want to synchronize the catalog with Google Sheets? ' +
          'This will download the latest sheets, update Qdrant databases, ' +
          'and rebuild the classifier models in the background.'
        }
        onConfirm={() => {
          setConfirmOpen(false);
          syncCache();
        }}
        confirmText="Sync"
      />

      <CatalogSyncDiffDialog
        open={diffDialogOpen}
        onClose={() => setDiffDialogOpen(false)}
        diffData={diffData}
        onConfirmSync={() => {
          setDiffDialogOpen(false);
          setConfirmOpen(true);
        }}
      />
    </PageContainer>
  );
}
