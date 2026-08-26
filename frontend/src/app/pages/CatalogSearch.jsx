import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  InputAdornment,
  Stack,
  Button,
  Tabs,
  Tab,
  Collapse,
  CircularProgress,
  Table,
  TableCell,
  TableRow,
  TablePagination,
  TableSortLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Chip,
  Alert,
} from '@mui/material';
import { Search, Database, Tag, Key, Layers, Award, Map, RefreshCw, SlidersHorizontal } from 'lucide-react';
import { PageContainer, PageHeader, ConfirmDialog } from '../components/ui';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { searchCatalog, buildCatalogCache, checkCatalogSync } from '../api';
import { useSnackbar } from 'notistack';
import { useDebounce } from '../hooks/useDebounce';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

export default function CatalogSearch() {
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();

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

  // Page states
  const [domain, setDomain] = useState('market');
  const [dataset, setDataset] = useState('catalog');
  const [query, setQuery] = useState('');

  // Advanced filters state (only for 'catalog' dataset)
  const [showFilters, setShowFilters] = useState(false);
  const [priceMin, setPriceMin] = useState('');
  const [priceMax, setPriceMax] = useState('');
  const [region, setRegion] = useState('');
  const [category, setCategory] = useState('');
  const [gkContains, setGkContains] = useState('');
  const [brandFilter, setBrandFilter] = useState('');
  const [btFilter, setBtFilter] = useState('');

  // Pagination states
  const [paginationModel, setPaginationModel] = useState({
    page: 0,
    pageSize: 25,
  });

  // Sorting state
  const [sortModel, setSortModel] = useState([]);

  // Cache sync state
  const [isChecking, setIsChecking] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [diffData, setDiffData] = useState(null);
  const [diffDialogOpen, setDiffDialogOpen] = useState(false);

  const { data: catalogStats } = useQuery({
    queryKey: ['catalog-stats-catalog', domain],
    queryFn: () => searchCatalog({ dataset: 'catalog', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });
  const { data: gkStats } = useQuery({
    queryKey: ['catalog-stats-gk', domain],
    queryFn: () => searchCatalog({ dataset: 'gk', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });
  const { data: btStats } = useQuery({
    queryKey: ['catalog-stats-bt', domain],
    queryFn: () => searchCatalog({ dataset: 'bt', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });
  const { data: categoryStats } = useQuery({
    queryKey: ['catalog-stats-category', domain],
    queryFn: () => searchCatalog({ dataset: 'category', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });
  const { data: brandsStats } = useQuery({
    queryKey: ['catalog-stats-brands', domain],
    queryFn: () => searchCatalog({ dataset: 'brands', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });
  const { data: mapStats } = useQuery({
    queryKey: ['catalog-stats-map', domain],
    queryFn: () => searchCatalog({ dataset: 'bt_gk_map', domain, page: 1, page_size: 1 }),
    staleTime: 5 * 60 * 1000,
  });

  // Memoize active filters to keep a stable object reference
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

  // Debounce the entire active filters block
  const debouncedFilters = useDebounce(activeFilters, 500);

  // Main catalog query
  const { data = { results: [], total: 0 }, isLoading } = useQuery({
    queryKey: ['catalog', dataset, domain, debouncedFilters, paginationModel.page, paginationModel.pageSize, sortModel],
    queryFn: () => {
      const sort = sortModel[0];
      return searchCatalog({
        dataset,
        domain,
        query: debouncedFilters.query.trim() || undefined,
        page: paginationModel.page + 1,
        page_size: paginationModel.pageSize,
        ...(sort && { sort_by: sort.field, sort_order: sort.sort }),
        // Include advanced filters if dataset is 'catalog'
        ...(dataset === 'catalog' && {
          min_price: debouncedFilters.priceMin ? Number(debouncedFilters.priceMin) : undefined,
          max_price: debouncedFilters.priceMax ? Number(debouncedFilters.priceMax) : undefined,
          region: debouncedFilters.region || undefined,
          category: debouncedFilters.category || undefined,
          gk_contains: debouncedFilters.gkContains || undefined,
          brand: debouncedFilters.brandFilter || undefined,
          basictype: debouncedFilters.btFilter || undefined,
        }),
      });
    },
  });

  // Reset page when dataset, domain or query changes
  const handleDatasetChange = (newDataset) => {
    setDataset(newDataset);
    setPaginationModel((prev) => ({ ...prev, page: 0 }));
  };

  const handleDomainChange = (e, newDomain) => {
    if (newDomain) {
      setDomain(newDomain);
      setPaginationModel((prev) => ({ ...prev, page: 0 }));
    }
  };

  const handleCheckSync = async () => {
    setIsChecking(true);
    try {
      const res = await checkCatalogSync({ limit: 50 });
      setDiffData(res);
      setDiffDialogOpen(true);
      if (res.has_changes) {
        enqueueSnackbar('Google Sheets has pending updates!', { variant: 'info' });
      } else {
        enqueueSnackbar('Your local cache is fully in-sync with Google Sheets.', { variant: 'success' });
      }
    } catch (err) {
      enqueueSnackbar(`Check failed: ${err.message || err}`, { variant: 'error' });
    } finally {
      setIsChecking(false);
    }
  };

  const handleSyncCache = async () => {
    setIsSyncing(true);
    try {
      await buildCatalogCache();
      enqueueSnackbar('Background cache build and pre-training initiated. Updates will apply shortly.', {
        variant: 'success',
      });
      // Invalidate queries so stats reload
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['catalog'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-catalog'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-gk'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-bt'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-category'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-brands'] });
        queryClient.invalidateQueries({ queryKey: ['catalog-stats-map'] });
      }, 5000);
    } catch (err) {
      enqueueSnackbar(`Sync failed: ${err.message || err}`, { variant: 'error' });
    } finally {
      setIsSyncing(false);
    }
  };

  // Define sidebar menu options
  const datasetsList = [
    { id: 'catalog', label: 'Catalog SKUs', icon: <Database size={18} /> },
    { id: 'gk', label: 'Generic Keywords', icon: <Tag size={18} /> },
    { id: 'bt', label: 'Basic Type', icon: <Key size={18} /> },
    { id: 'category', label: domain === 'food' ? 'Region' : 'Categories', icon: <Layers size={18} /> },
    { id: 'brands', label: domain === 'food' ? 'Flavors List' : 'Brands List', icon: <Award size={18} /> },
    { id: 'bt_gk_map', label: 'Basic Type - Generic Keywords Map', icon: <Map size={18} /> },
  ];

  // Memoize column definitions — avoids recreating arrays on every render cycle
  const columns = useMemo(() => {
    switch (dataset) {
      case 'catalog':
        if (domain === 'market') {
          return [
            { field: 'name', headerName: 'SKU Name', flex: 1.5, minWidth: 260 },
            { field: 'category', headerName: 'Categories', width: 140 },
            { field: 'gk', headerName: 'Generic Keywords', flex: 1.2, minWidth: 250 },
            { field: 'bt', headerName: 'Basic Type', width: 150 },
            { field: 'brand', headerName: 'Brand', width: 140 },
            {
              field: 'price',
              headerName: 'Price',
              width: 100,
              valueFormatter: (value) => {
                if (value == null) return '';
                const num = Number(value);
                return isNaN(num) ? String(value) : `${num.toFixed(2)}`;
              },
            },
            { field: 'merchant', headerName: 'Merchant', width: 130 },
          ];
        } else {
          return [
            { field: 'name', headerName: 'SKU Name', flex: 1.5, minWidth: 260 },
            { field: 'gk', headerName: 'Generic Keywords', flex: 1.2, minWidth: 250 },
            { field: 'bt', headerName: 'Basic Type', width: 150 },
            { field: 'region', headerName: 'Region', width: 120 },
            { field: 'flavor', headerName: 'Flavor', width: 140 },
            {
              field: 'price',
              headerName: 'Price',
              width: 100,
              valueFormatter: (value) => {
                if (value == null) return '';
                const num = Number(value);
                return isNaN(num) ? String(value) : `${num.toFixed(2)}`;
              },
            },
            { field: 'description', headerName: 'Description', flex: 1, minWidth: 200 },
          ];
        }
      case 'gk':
        return [
          { field: 'name', headerName: 'Generic Keywords Tag', flex: 1, minWidth: 250 },
          {
            field: 'count',
            headerName: 'Occurrence SKU Count',
            width: 180,
            align: 'right',
            headerAlign: 'right',
            valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
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
            valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
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
            valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
          },
        ];
      case 'brands':
        if (domain === 'market') {
          return [
            { field: 'name', headerName: 'Brand Name', flex: 1, minWidth: 180 },
            { field: 'aliases', headerName: 'Aliases', flex: 1, minWidth: 200 },
            {
              field: 'is_weak',
              headerName: 'Weak Brand?',
              width: 130,
              renderCell: (params) => (params.value ? 'Yes' : 'No'),
            },
            {
              field: 'count',
              headerName: 'Occurrence SKU Count',
              width: 180,
              align: 'right',
              headerAlign: 'right',
              valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
            },
          ];
        } else {
          return [
            { field: 'name', headerName: 'Flavor Name', flex: 1, minWidth: 150 },
            { field: 'aliases', headerName: 'Aliases', flex: 1, minWidth: 200 },
            {
              field: 'is_meat',
              headerName: 'Is Meat?',
              width: 110,
              renderCell: (params) => (params.value ? 'Yes' : 'No'),
            },
            {
              field: 'is_vegetable',
              headerName: 'Is Vegetable?',
              width: 135,
              renderCell: (params) => (params.value ? 'Yes' : 'No'),
            },
            {
              field: 'is_seafood',
              headerName: 'Is Seafood?',
              width: 130,
              renderCell: (params) => (params.value ? 'Yes' : 'No'),
            },
            {
              field: 'count',
              headerName: 'Occurrence SKU Count',
              width: 180,
              align: 'right',
              headerAlign: 'right',
              valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
            },
          ];
        }
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
            valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
          },
          {
            field: 'count',
            headerName: 'Associated SKU Count',
            width: 190,
            align: 'right',
            headerAlign: 'right',
            valueFormatter: (value) => (value != null ? value.toLocaleString() : '0'),
          },
        ];
      default:
        return [];
    }
  }, [dataset, domain]);

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
              startIcon={isChecking ? <CircularProgress size={16} color="inherit" /> : <Search size={16} />}
              disabled={isChecking || isSyncing}
              onClick={handleCheckSync}
            >
              {isChecking ? 'Checking...' : 'Check for Updates'}
            </Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={isSyncing ? <CircularProgress size={16} color="inherit" /> : <RefreshCw size={16} />}
              disabled={isSyncing}
              onClick={() => setConfirmOpen(true)}
            >
              {isSyncing ? 'Syncing...' : 'Sync from Sheets'}
            </Button>
          </Stack>
        }
      />

      {/* Domain Switcher */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={domain} onChange={handleDomainChange} indicatorColor="primary" textColor="primary">
          <Tab label="Market Domain" value="market" sx={{ fontWeight: 600, textTransform: 'capitalize' }} />
          <Tab label="Food Domain" value="food" sx={{ fontWeight: 600, textTransform: 'capitalize' }} />
        </Tabs>
      </Box>

      {/* Overview Metrics Cards */}
      <Box
        sx={{
          display: 'flex',
          width: '100%',
          gap: 2,
          mb: 4,
          flexWrap: 'wrap',
          '& > *': {
            flex: '1 1 0px',
            minWidth: { xs: 'calc(50% - 16px)', sm: 'calc(33.33% - 16px)', md: '0px' },
          },
        }}
      >
        {/* Card 1: Catalog SKUs */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'catalog' ? '2px solid' : '1px solid',
            borderColor: dataset === 'catalog' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'catalog' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('catalog')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              Catalog SKUs
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {catalogStats?.total != null ? catalogStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>

        {/* Card 2: Generic keywords */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'gk' ? '2px solid' : '1px solid',
            borderColor: dataset === 'gk' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'gk' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('gk')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              Generic Keywords
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {gkStats?.total != null ? gkStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>

        {/* Card 3: Basic type */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'bt' ? '2px solid' : '1px solid',
            borderColor: dataset === 'bt' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'bt' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('bt')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              Basic Type
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {btStats?.total != null ? btStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>

        {/* Card 4: Category / Region */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'category' ? '2px solid' : '1px solid',
            borderColor: dataset === 'category' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'category' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('category')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              {domain === 'food' ? 'Region' : 'Categories'}
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {categoryStats?.total != null ? categoryStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>

        {/* Card 5: Brands / Flavors */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'brands' ? '2px solid' : '1px solid',
            borderColor: dataset === 'brands' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'brands' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('brands')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              {domain === 'food' ? 'Flavors List' : 'Brands List'}
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {brandsStats?.total != null ? brandsStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>

        {/* Card 6: Basic type - Generic keywords Map */}
        <Card
          sx={{
            cursor: 'pointer',
            border: dataset === 'bt_gk_map' ? '2px solid' : '1px solid',
            borderColor: dataset === 'bt_gk_map' ? 'primary.main' : 'divider',
            bgcolor: dataset === 'bt_gk_map' ? 'action.selected' : 'background.paper',
            '&:hover': { borderColor: 'primary.main' },
          }}
          onClick={() => handleDatasetChange('bt_gk_map')}
        >
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              Basic Type - Generic Keywords Map
            </Typography>
            <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
              {mapStats?.total != null ? mapStats.total.toLocaleString() : '---'}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Main Exploration Section */}
      <Grid container spacing={3}>
        {/* Search Results and Filters Panel */}
        <Grid item xs={12}>
          {/* Filters Card */}
          <Card sx={{ p: 2, mb: 3 }}>
            <Stack direction="row" spacing={2} sx={{ width: '100%' }}>
              <TextField
                size="small"
                placeholder={`Search ${datasetsList.find((d) => d.id === dataset)?.label || 'dataset'}...`}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setPaginationModel((prev) => ({ ...prev, page: 0 }));
                }}
                sx={{ flexGrow: 1 }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search size={16} />
                    </InputAdornment>
                  ),
                }}
              />

              {dataset === 'catalog' && (
                <Button
                  variant="outlined"
                  color={showFilters ? 'primary' : 'inherit'}
                  startIcon={<SlidersHorizontal size={16} />}
                  onClick={() => setShowFilters(!showFilters)}
                >
                  Filters
                </Button>
              )}
            </Stack>

            {/* Advanced Filters Expandable block */}
            <Collapse in={showFilters && dataset === 'catalog'} sx={{ mt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4} md={3}>
                  <TextField
                    size="small"
                    fullWidth
                    label="Min Price"
                    type="number"
                    value={priceMin}
                    onChange={(e) => {
                      setPriceMin(e.target.value);
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4} md={3}>
                  <TextField
                    size="small"
                    fullWidth
                    label="Max Price"
                    type="number"
                    value={priceMax}
                    onChange={(e) => {
                      setPriceMax(e.target.value);
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  />
                </Grid>
                {domain === 'food' ? (
                  <Grid item xs={12} sm={4} md={3}>
                    <TextField
                      size="small"
                      fullWidth
                      label="Region"
                      value={region}
                      onChange={(e) => {
                        setRegion(e.target.value);
                        setPaginationModel((prev) => ({ ...prev, page: 0 }));
                      }}
                    />
                  </Grid>
                ) : (
                  <Grid item xs={12} sm={4} md={3}>
                    <TextField
                      size="small"
                      fullWidth
                      label="Categories"
                      value={category}
                      onChange={(e) => {
                        setCategory(e.target.value);
                        setPaginationModel((prev) => ({ ...prev, page: 0 }));
                      }}
                    />
                  </Grid>
                )}
                <Grid item xs={12} sm={4} md={3}>
                  <TextField
                    size="small"
                    fullWidth
                    label="Generic Keywords"
                    value={gkContains}
                    onChange={(e) => {
                      setGkContains(e.target.value);
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4} md={3}>
                  <TextField
                    size="small"
                    fullWidth
                    label={domain === 'food' ? 'Flavor' : 'Brand'}
                    value={brandFilter}
                    onChange={(e) => {
                      setBrandFilter(e.target.value);
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4} md={3}>
                  <TextField
                    size="small"
                    fullWidth
                    label="Basic Type"
                    value={btFilter}
                    onChange={(e) => {
                      setBtFilter(e.target.value);
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4} md={3}>
                  <Button
                    fullWidth
                    variant="text"
                    color="secondary"
                    onClick={() => {
                      setPriceMin('');
                      setPriceMax('');
                      setRegion('');
                      setCategory('');
                      setGkContains('');
                      setBrandFilter('');
                      setBtFilter('');
                      setPaginationModel((prev) => ({ ...prev, page: 0 }));
                    }}
                  >
                    Clear Filters
                  </Button>
                </Grid>
              </Grid>
            </Collapse>
          </Card>

          {/* Interactive Data Table Card */}
          <Box sx={{ mb: 3 }}>
            <StyledTableContainer>
              <Table aria-label="catalog table">
                <StyledTableHead>
                  <TableRow>
                    {columns.map((col) => {
                      const currentSort = sortModel[0];
                      const isSorted = currentSort && currentSort.field === col.field;
                      const sortDirection = isSorted ? currentSort.sort : undefined;
                      return (
                        <StyledHeaderCell
                          key={col.field}
                          align={col.align || col.headerAlign || 'left'}
                          sx={{
                            cursor: col.sortable !== false ? 'pointer' : 'default',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {col.sortable !== false ? (
                            <TableSortLabel
                              active={!!isSorted}
                              direction={sortDirection || 'asc'}
                              onClick={() => handleHeaderClick(col)}
                            >
                              {col.headerName}
                            </TableSortLabel>
                          ) : (
                            col.headerName
                          )}
                        </StyledHeaderCell>
                      );
                    })}
                  </TableRow>
                </StyledTableHead>
                <StyledTableBody>
                  {isLoading ? (
                    <TableSkeleton columns={columns.length} rows={5} />
                  ) : (data?.results || []).length === 0 ? (
                    <StyledTableRow>
                      <TableCell colSpan={columns.length} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No records found.
                        </Typography>
                      </TableCell>
                    </StyledTableRow>
                  ) : (
                    (data?.results || []).map((row) => (
                      <StyledTableRow key={row.id}>
                        {columns.map((col) => {
                          const value = row[col.field];
                          let cellContent = value;
                          if (col.renderCell) {
                            cellContent = col.renderCell({ value, row });
                          } else if (col.valueFormatter) {
                            cellContent = col.valueFormatter(value, row);
                          }
                          return (
                            <TableCell key={col.field} align={col.align || 'left'}>
                              {cellContent}
                            </TableCell>
                          );
                        })}
                      </StyledTableRow>
                    ))
                  )}
                </StyledTableBody>
              </Table>
              <TablePagination
                rowsPerPageOptions={[10, 25, 50, 100]}
                component="div"
                count={data?.total || 0}
                rowsPerPage={paginationModel.pageSize}
                page={paginationModel.page}
                onPageChange={(e, newPage) => setPaginationModel((prev) => ({ ...prev, page: newPage }))}
                onRowsPerPageChange={(e) =>
                  setPaginationModel((prev) => ({
                    ...prev,
                    pageSize: parseInt(e.target.value, 10),
                    page: 0,
                  }))
                }
                sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
              />
            </StyledTableContainer>
          </Box>
        </Grid>
      </Grid>

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title="Sync from Google Sheets"
        message="Are you sure you want to synchronize the catalog with Google Sheets? This will download the latest sheets, update Meilisearch/Qdrant databases, and rebuild the classifier models in the background."
        onConfirm={() => {
          setConfirmOpen(false);
          handleSyncCache();
        }}
        confirmText="Sync"
      />

      <Dialog
        open={diffDialogOpen}
        onClose={() => setDiffDialogOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Database size={20} /> Google Sheets Sync Status
        </DialogTitle>
        <DialogContent dividers>
          {diffData && (
            <Stack spacing={2}>
              {!diffData.has_changes ? (
                <Box sx={{ py: 3, textAlign: 'center' }}>
                  <Typography variant="h6" color="success.main" sx={{ fontWeight: 600, mb: 1 }}>
                    ✓ Catalog is Up to Date
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    All local caches match the Google Sheets data perfectly. No synchronization is required.
                  </Typography>
                </Box>
              ) : (
                <Box>
                  {(() => {
                    const market = diffData.details?.market || {
                      new_count: 0,
                      changed_count: 0,
                      new_rows: [],
                      changed_rows: [],
                    };
                    const food = diffData.details?.food || {
                      new_count: 0,
                      changed_count: 0,
                      new_rows: [],
                      changed_rows: [],
                    };
                    const totalNew = market.new_count + food.new_count;
                    const totalChanged = market.changed_count + food.changed_count;
                    const grandTotal = totalNew + totalChanged;
                    const isCapped =
                      market.is_capped ||
                      food.is_capped ||
                      market.new_count > market.new_rows?.length ||
                      market.changed_count > market.changed_rows?.length ||
                      food.new_count > food.new_rows?.length ||
                      food.changed_count > food.changed_rows?.length;

                    return (
                      <>
                        <Alert severity="warning" sx={{ mb: 2.5 }}>
                          Found <strong>{grandTotal.toLocaleString()}</strong> total pending catalog updates (
                          <strong>{totalNew.toLocaleString()}</strong> new rows,{' '}
                          <strong>{totalChanged.toLocaleString()}</strong> modified rows).
                          {isCapped &&
                            ' Displaying top preview rows below. Synchronizing will apply all updates in the background.'}
                        </Alert>

                        {['market', 'food'].map((domainKey) => {
                          const domainDetails = diffData.details?.[domainKey] || {
                            new_count: 0,
                            changed_count: 0,
                            new_rows: [],
                            changed_rows: [],
                          };
                          if (domainDetails.new_count === 0 && domainDetails.changed_count === 0) return null;

                          // Slice to top 20 for list rendering safety
                          const previewNew = (domainDetails.new_rows || []).slice(0, 20);
                          const previewChanged = (domainDetails.changed_rows || []).slice(0, 20);
                          const remainingNew = domainDetails.new_count - previewNew.length;
                          const remainingChanged = domainDetails.changed_count - previewChanged.length;

                          return (
                            <Card variant="outlined" key={domainKey} sx={{ mb: 2, borderRadius: '8px' }}>
                              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1.5 }}>
                                  <Typography
                                    variant="subtitle2"
                                    sx={{ fontWeight: 700, textTransform: 'uppercase', color: 'primary.main' }}
                                  >
                                    {domainKey === 'market' ? 'Market Domain' : 'Food Domain'}
                                  </Typography>
                                  <Stack direction="row" spacing={1}>
                                    {domainDetails.new_count > 0 && (
                                      <Chip
                                        size="small"
                                        label={`${domainDetails.new_count.toLocaleString()} New`}
                                        color="success"
                                        variant="outlined"
                                        sx={{ fontWeight: 600 }}
                                      />
                                    )}
                                    {domainDetails.changed_count > 0 && (
                                      <Chip
                                        size="small"
                                        label={`${domainDetails.changed_count.toLocaleString()} Modified`}
                                        color="warning"
                                        variant="outlined"
                                        sx={{ fontWeight: 600 }}
                                      />
                                    )}
                                  </Stack>
                                </Stack>

                                {previewNew.length > 0 && (
                                  <Box sx={{ mt: 1.5 }}>
                                    <Typography
                                      variant="caption"
                                      display="block"
                                      sx={{ fontWeight: 700, color: 'success.main', mb: 0.5 }}
                                    >
                                      NEW SKUs (showing {previewNew.length} of{' '}
                                      {domainDetails.new_count.toLocaleString()}):
                                    </Typography>
                                    <List
                                      size="small"
                                      dense
                                      sx={{ bgcolor: 'action.hover', borderRadius: '6px', py: 0.5 }}
                                    >
                                      {previewNew.map((row, i) => (
                                        <ListItem key={i} sx={{ py: 0.2 }}>
                                          <ListItemText
                                            primary={row.name}
                                            primaryTypographyProps={{ variant: 'body2', sx: { fontWeight: 500 } }}
                                            secondary={
                                              row.details.basictype ? `Basic Type: ${row.details.basictype}` : null
                                            }
                                          />
                                        </ListItem>
                                      ))}
                                    </List>
                                    {remainingNew > 0 && (
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ display: 'block', mt: 0.5, fontStyle: 'italic', pl: 1 }}
                                      >
                                        + {remainingNew.toLocaleString()} more new rows will be added during sync.
                                      </Typography>
                                    )}
                                  </Box>
                                )}

                                {previewChanged.length > 0 && (
                                  <Box sx={{ mt: 1.5 }}>
                                    <Typography
                                      variant="caption"
                                      display="block"
                                      sx={{ fontWeight: 700, color: 'warning.main', mb: 0.5 }}
                                    >
                                      MODIFIED SKUs (showing {previewChanged.length} of{' '}
                                      {domainDetails.changed_count.toLocaleString()}):
                                    </Typography>
                                    <List
                                      size="small"
                                      dense
                                      sx={{ bgcolor: 'action.hover', borderRadius: '6px', py: 0.5 }}
                                    >
                                      {previewChanged.map((row, i) => (
                                        <ListItem key={i} sx={{ py: 0.5, alignItems: 'flex-start' }}>
                                          <Box sx={{ width: '100%' }}>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                              {row.name}
                                            </Typography>
                                            {Object.entries(row.diffs).map(([field, delta], j) => (
                                              <Typography
                                                key={j}
                                                variant="caption"
                                                display="block"
                                                color="text.secondary"
                                                sx={{ pl: 1 }}
                                              >
                                                • {field}:{' '}
                                                <span
                                                  style={{
                                                    textDecoration: 'line-through',
                                                    color: '#ff8a8a',
                                                    paddingRight: '4px',
                                                  }}
                                                >
                                                  '{delta.old}'
                                                </span>{' '}
                                                →{' '}
                                                <span style={{ color: '#8aff8a', fontWeight: 600 }}>'{delta.new}'</span>
                                              </Typography>
                                            ))}
                                          </Box>
                                        </ListItem>
                                      ))}
                                    </List>
                                    {remainingChanged > 0 && (
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ display: 'block', mt: 0.5, fontStyle: 'italic', pl: 1 }}
                                      >
                                        + {remainingChanged.toLocaleString()} more modified rows will be updated during
                                        sync.
                                      </Typography>
                                    )}
                                  </Box>
                                )}
                              </CardContent>
                            </Card>
                          );
                        })}
                      </>
                    );
                  })()}
                </Box>
              )}
            </Stack>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setDiffDialogOpen(false)} color="inherit">
            Close
          </Button>
          {diffData?.has_changes && (
            <Button
              variant="contained"
              color="primary"
              onClick={() => {
                setDiffDialogOpen(false);
                setConfirmOpen(true);
              }}
            >
              Sync from Sheets
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </PageContainer>
  );
}
