import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  Button,
  MenuItem,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Table,
  TableRow,
  TableCell,
} from '@mui/material';
import { Search, Circle, Info } from 'lucide-react';
import prettyBytes from 'pretty-bytes';
import { PageContainer, PageHeader, SideDrawer } from '../components/ui';
import { useQuery, useMutation } from '@tanstack/react-query';
import { getCollections, getCollection, searchCollection, runInteractiveAudit } from '../api';
import { useSnackbar } from 'notistack';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

const statusColor = { green: 'success.main', yellow: 'warning.main', red: 'error.main' };

const dictTypeLabels = {
  gk: 'Generic Keywords',
  bt: 'Basic Type',
  region: 'Region',
  category: 'Category',
};

// Collections backed by the SKU matcher's hybrid retrieval + RRF + cross-encoder pipeline.
const COLLECTION_DOMAINS = {
  market_catalog: 'market',
  food_catalog: 'food',
};

export default function Collections() {
  const { enqueueSnackbar } = useSnackbar();
  const [collection, setCollection] = useState('');
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState('dense');
  const [results, setResults] = useState(null);
  const [detailsDrawer, setDetailsDrawer] = useState(null);
  const [payloadDrawer, setPayloadDrawer] = useState(null);

  const matcherDomain = COLLECTION_DOMAINS[collection] || null;

  // Auto-switch away from pipeline mode if selected collection has no pipeline support
  React.useEffect(() => {
    if (!matcherDomain && searchMode === 'pipeline') {
      setSearchMode('dense');
    }
  }, [matcherDomain, searchMode]);

  const { data: collections = [], isLoading } = useQuery({
    queryKey: ['collections'],
    queryFn: async () => {
      const data = await getCollections();
      const details = await Promise.all(
        data.collections.map(async (name) => {
          try {
            const info = await getCollection(name);
            const size = info.config?.params?.vectors?.dense?.size || info.config?.params?.vectors?.size || 1024;
            const dist =
              info.config?.params?.vectors?.dense?.distance || info.config?.params?.vectors?.distance || 'Cosine';
            return {
              name,
              status: info.status || 'unknown',
              vectors_count: info.vectors_count || 0,
              points_count: info.points_count || 0,
              segments_count: info.segments_count || 0,
              vector_size: size,
              distance: dist,
              disk_bytes: info.optimizer_status?.disk_bytes || 0,
              ram_bytes: info.optimizer_status?.ram_bytes || 0,
              shards_count: 1,
              vectors_config: `dense, ${size}, ${dist}`,
            };
          } catch (e) {
            return { name, status: 'error', vectors_count: 0, points_count: 0, segments_count: 0 };
          }
        })
      );
      return details;
    },
  });

  // Set default collection once loaded
  React.useEffect(() => {
    if (collections.length > 0 && !collection) {
      setCollection(collections[0].name);
    }
  }, [collections, collection]);

  // Reset results when collection or mode changes
  React.useEffect(() => {
    setResults(null);
  }, [collection, searchMode]);

  // Converts a matcher audit trace's cross-encoder candidate pool into the same
  // { id, score, payload } shape the dense-search results table already renders.
  const auditToResults = (auditData) =>
    (auditData?.stage3_cross_encoder?.top_candidates || []).map((c, idx) => ({
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

  const searchMutation = useMutation({
    mutationFn: searchCollection,
    onSuccess: (data) => setResults(data.results),
    onError: (e) => {
      enqueueSnackbar(`Search failed: ${e.message}`, { variant: 'error' });
    },
  });

  const auditMutation = useMutation({
    mutationFn: runInteractiveAudit,
    onSuccess: (data) => setResults(auditToResults(data)),
    onError: (e) => {
      enqueueSnackbar(`Matcher pipeline search failed: ${e.message}`, { variant: 'error' });
    },
  });

  const runSearch = () => {
    if (!query.trim() || !collection) {
      setResults([]);
      return;
    }
    if (searchMode === 'pipeline') {
      if (matcherDomain) {
        auditMutation.mutate({ sku_name: query.trim(), domain: matcherDomain, task: 'matcher' });
      } else {
        enqueueSnackbar('Pipeline search is only available for catalog collections.', { variant: 'warning' });
      }
    } else {
      searchMutation.mutate({
        name: collection,
        query: query.trim(),
        topK: 10,
        mode: searchMode,
        scoreThreshold: searchMode === 'dense' ? 0.1 : undefined,
      });
    }
  };

  const isSearching = searchMutation.isPending || auditMutation.isPending;

  return (
    <PageContainer>
      <PageHeader
        title="Collections"
        subtitle="Qdrant collections backing the matcher, with live stats and a vector search box."
      />

      <StyledTableContainer>
        <Table aria-label="collections table">
          <StyledTableHead>
            <TableRow>
              <StyledHeaderCell width="25%">Name</StyledHeaderCell>
              <StyledHeaderCell width="15%">Status</StyledHeaderCell>
              <StyledHeaderCell width="15%">Points (Approx)</StyledHeaderCell>
              <StyledHeaderCell width="10%">Segments</StyledHeaderCell>
              <StyledHeaderCell width="10%">Shards</StyledHeaderCell>
              <StyledHeaderCell width="30%">Vectors Config</StyledHeaderCell>
              <StyledHeaderCell align="right">Actions</StyledHeaderCell>
            </TableRow>
          </StyledTableHead>
          <StyledTableBody>
            {isLoading ? (
              <TableSkeleton columns={7} rows={5} />
            ) : collections.length === 0 ? (
              <StyledTableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No collections found.
                  </Typography>
                </TableCell>
              </StyledTableRow>
            ) : (
              collections.map((row) => (
                <StyledTableRow key={row.name}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Circle
                        size={10}
                        fill="currentColor"
                        color={statusColor[row.status] ? 'inherit' : undefined}
                        style={{
                          color: row.status === 'green' ? '#2e7d32' : row.status === 'yellow' ? '#ed6c02' : '#d32f2f',
                        }}
                      />
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {row.name}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      label={row.status.toUpperCase()}
                      sx={{
                        color: row.status === 'green' ? '#2e7d32' : row.status === 'yellow' ? '#ed6c02' : '#d32f2f',
                        fontWeight: 600,
                        borderColor:
                          row.status === 'green' ? '#2e7d32' : row.status === 'yellow' ? '#ed6c02' : '#d32f2f',
                      }}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>{row.points_count?.toLocaleString()}</TableCell>
                  <TableCell>{row.segments_count}</TableCell>
                  <TableCell>{row.shards_count}</TableCell>
                  <TableCell>{row.vectors_config}</TableCell>
                  <TableCell align="right">
                    <IconButton
                      aria-label="View collection details"
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDetailsDrawer(row);
                      }}
                    >
                      <Info size={18} />
                    </IconButton>
                  </TableCell>
                </StyledTableRow>
              ))
            )}
          </StyledTableBody>
        </Table>
      </StyledTableContainer>

      <Card sx={{ mt: 3 }}>
        <CardHeader title="Vector search" titleTypographyProps={{ variant: 'subtitle1' }} />
        <CardContent>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
            <TextField
              select
              size="small"
              label="Collection"
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              sx={{ width: 200 }}
            >
              {collections.map((c) => (
                <MenuItem key={c.name} value={c.name}>
                  {c.name}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              size="small"
              label="Mode"
              value={searchMode}
              onChange={(e) => setSearchMode(e.target.value)}
              sx={{ width: 140 }}
            >
              <MenuItem value="dense">Dense</MenuItem>
              <MenuItem value="sparse">Sparse</MenuItem>
              <MenuItem value="hybrid">Hybrid</MenuItem>
              <MenuItem value="pipeline" disabled={!matcherDomain}>
                Pipeline {!matcherDomain && '(Catalog only)'}
              </MenuItem>
            </TextField>
            <TextField
              size="small"
              placeholder="e.g. coca cola 330ml can"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runSearch()}
              sx={{ flexGrow: 1, minWidth: 220 }}
            />
            <Button variant="contained" disabled={isSearching} startIcon={<Search size={16} />} onClick={runSearch}>
              Search
            </Button>
          </Box>

          {results !== null && (
            <>
              <Divider sx={{ my: 2 }} />
              {results.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No results. Enter a query above.
                </Typography>
              ) : (
                <StyledTableContainer sx={{ mt: 2 }}>
                  <Table size="small">
                    <StyledTableHead>
                      <TableRow>
                        {collection.endsWith('_tags') ? (
                          <>
                            <StyledHeaderCell width="45%">Tag</StyledHeaderCell>
                            <StyledHeaderCell width="30%">Dict Type</StyledHeaderCell>
                          </>
                        ) : (
                          <>
                            <StyledHeaderCell width="25%">Name</StyledHeaderCell>
                            <StyledHeaderCell width="40%">Description</StyledHeaderCell>
                            <StyledHeaderCell width="20%">Seller Category</StyledHeaderCell>
                          </>
                        )}
                        <StyledHeaderCell width="10%">
                          {searchMode === 'pipeline' ? 'Raw CE Score' : 'Score'}
                        </StyledHeaderCell>
                        <StyledHeaderCell width="5%" align="right">
                          Payload
                        </StyledHeaderCell>
                      </TableRow>
                    </StyledTableHead>
                    <StyledTableBody>
                      {results.map((r) => {
                        const name =
                          r.payload?.Name || r.payload?.name || r.payload?.catalog_name || r.payload?.tag || 'N/A';
                        const desc = r.payload?.Description || r.payload?.description || '—';
                        const sellerCategory =
                          r.payload?.SellerCategory ||
                          r.payload?.['Seller Category'] ||
                          r.payload?.Seller_Category ||
                          r.payload?.category ||
                          r.payload?.Category ||
                          r.payload?.region ||
                          r.payload?.Region ||
                          '—';

                        // Tag specific fields
                        const dictType = r.payload?.dict_type || 'N/A';
                        const dictTypeDisplay = dictTypeLabels[dictType] || dictType;

                        const scoreVal = r.score ?? r.raw_cross_score;
                        const scoreFormatted =
                          scoreVal !== undefined && scoreVal !== null ? Number(scoreVal).toFixed(4) : 'N/A';

                        const scoreColor =
                          searchMode === 'pipeline'
                            ? scoreVal >= 5
                              ? 'success'
                              : scoreVal >= 0
                              ? 'primary'
                              : 'default'
                            : searchMode === 'hybrid'
                            ? scoreVal >= 0.015
                              ? 'success'
                              : 'primary'
                            : scoreVal >= 0.8
                            ? 'success'
                            : scoreVal >= 0.5
                            ? 'primary'
                            : 'default';

                        return (
                          <StyledTableRow key={r.id}>
                            {collection.endsWith('_tags') ? (
                              <>
                                <TableCell sx={{ fontWeight: 500 }}>{name}</TableCell>
                                <TableCell>{dictTypeDisplay}</TableCell>
                              </>
                            ) : (
                              <>
                                <TableCell sx={{ fontWeight: 500 }}>{name}</TableCell>
                                <TableCell sx={{ color: 'text.secondary', fontSize: '0.825rem' }}>{desc}</TableCell>
                                <TableCell>{sellerCategory}</TableCell>
                              </>
                            )}
                            <TableCell>
                              <Chip size="small" label={scoreFormatted} color={scoreColor} variant="outlined" />
                            </TableCell>
                            <TableCell align="right">
                              <IconButton
                                size="small"
                                onClick={() => setPayloadDrawer(r)}
                                aria-label="View raw payload"
                              >
                                <Info size={16} />
                              </IconButton>
                            </TableCell>
                          </StyledTableRow>
                        );
                      })}
                    </StyledTableBody>
                  </Table>
                </StyledTableContainer>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <SideDrawer open={!!detailsDrawer} onClose={() => setDetailsDrawer(null)} title={detailsDrawer?.name} width={400}>
        {detailsDrawer && (
          <List>
            <ListItem disableGutters>
              <ListItemText primary="Status" secondary={detailsDrawer.status} />
            </ListItem>
            <Divider />
            <ListItem disableGutters>
              <ListItemText primary="Points (approx)" secondary={detailsDrawer.points_count?.toLocaleString()} />
            </ListItem>
            <Divider />
            <ListItem disableGutters>
              <ListItemText primary="Segments" secondary={detailsDrawer.segments_count} />
            </ListItem>
            <Divider />
            <ListItem disableGutters>
              <ListItemText primary="Vectors configuration" secondary={detailsDrawer.vectors_config} />
            </ListItem>
            <Divider />
            <ListItem disableGutters>
              <ListItemText primary="RAM usage" secondary={prettyBytes(detailsDrawer.ram_bytes)} />
            </ListItem>
            <Divider />
            <ListItem disableGutters>
              <ListItemText primary="Disk usage" secondary={prettyBytes(detailsDrawer.disk_bytes)} />
            </ListItem>
          </List>
        )}
      </SideDrawer>

      <SideDrawer open={!!payloadDrawer} onClose={() => setPayloadDrawer(null)} title="Point Details" width={500}>
        {payloadDrawer && (
          <Box>
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Point ID
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                {payloadDrawer.id}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Chip size="small" label={`Score: ${payloadDrawer.score.toFixed(6)}`} color="primary" />
              </Box>
            </Box>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
              Payload
            </Typography>
            <Box
              sx={{ flexGrow: 1, overflow: 'auto', bgcolor: 'grey.900', color: 'common.white', p: 2, borderRadius: 1 }}
            >
              <pre
                style={{
                  margin: 0,
                  fontFamily: 'monospace',
                  fontSize: '0.825rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {JSON.stringify(payloadDrawer.payload, null, 2)}
              </pre>
            </Box>
          </Box>
        )}
      </SideDrawer>
    </PageContainer>
  );
}
