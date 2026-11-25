import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  IconButton,
  MenuItem,
  Table,
  TableCell,
  TableRow,
  TextField,
} from '@mui/material';
import { InfoOutlined, Search } from '@mui/icons-material';
import { DICT_TYPE_LABELS } from '../../utils/constants';
import { EmptyState } from '../common/EmptyState';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';

function resolveScoreColor(searchMode, scoreVal) {
  if (searchMode === 'pipeline') {
    if (scoreVal >= 5) return 'success';
    if (scoreVal >= 0) return 'primary';
    return 'default';
  }
  if (searchMode === 'hybrid') {
    return scoreVal >= 0.015 ? 'success' : 'primary';
  }
  if (scoreVal >= 0.8) return 'success';
  if (scoreVal >= 0.5) return 'primary';
  return 'default';
}

export function VectorSearchPanel({
  collections,
  collection,
  onCollectionChange,
  searchMode,
  onSearchModeChange,
  matcherDomain,
  query,
  onQueryChange,
  isSearching,
  onRunSearch,
  results,
  onSelectPayload,
}) {
  return (
    <Card sx={{ mt: 3 }}>
      <CardHeader title="Vector search" titleTypographyProps={{ variant: 'subtitle1' }} />
      <CardContent>
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField
            select
            size="small"
            label="Collection"
            value={collection}
            onChange={(e) => onCollectionChange(e.target.value)}
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
            onChange={(e) => onSearchModeChange(e.target.value)}
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
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onRunSearch()}
            sx={{ flexGrow: 1, minWidth: 220 }}
          />
          <Button
            variant="contained"
            disabled={isSearching}
            startIcon={<Search sx={{ fontSize: 16 }} />}
            onClick={onRunSearch}
          >
            Search
          </Button>
        </Box>

        {results !== null && (
          <>
            <Divider sx={{ my: 2 }} />
            {results.length === 0 ? (
              <EmptyState
                icon={Search}
                title="No search results"
                description="No vector items matched your test query. Try adjusting query terms or score threshold."
                minHeight={180}
              />
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
                      const dictType = r.payload?.dict_type || 'N/A';
                      const dictTypeDisplay = DICT_TYPE_LABELS[dictType] || dictType;

                      const scoreVal = r.score ?? r.raw_cross_score;
                      const scoreFormatted =
                        scoreVal !== undefined && scoreVal !== null ? Number(scoreVal).toFixed(4) : 'N/A';

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
                            <Chip
                              size="small"
                              label={scoreFormatted}
                              color={resolveScoreColor(searchMode, scoreVal)}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell align="right">
                            <IconButton size="small" onClick={() => onSelectPayload(r)} aria-label="View raw payload">
                              <InfoOutlined sx={{ fontSize: 16 }} />
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
  );
}

VectorSearchPanel.propTypes = {
  collections: PropTypes.array.isRequired,
  collection: PropTypes.string.isRequired,
  onCollectionChange: PropTypes.func.isRequired,
  searchMode: PropTypes.string.isRequired,
  onSearchModeChange: PropTypes.func.isRequired,
  matcherDomain: PropTypes.string,
  query: PropTypes.string.isRequired,
  onQueryChange: PropTypes.func.isRequired,
  isSearching: PropTypes.bool.isRequired,
  onRunSearch: PropTypes.func.isRequired,
  results: PropTypes.array,
  onSelectPayload: PropTypes.func.isRequired,
};
