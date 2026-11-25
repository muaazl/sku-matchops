import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, IconButton, Table, TableCell, TableRow, Typography } from '@mui/material';
import { FiberManualRecord, InfoOutlined, Storage } from '@mui/icons-material';
import { COLLECTION_STATUS_COLORS } from '../../utils/constants';
import { EmptyState } from '../common/EmptyState';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function CollectionsTable({ collections, isLoading, onSelectCollection }) {
  return (
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
              <TableCell colSpan={7} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={Storage}
                  title="No vector collections found"
                  description="No collections were found in the connected vector store."
                  minHeight={200}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            collections.map((row) => {
              const badgeColor = row.status === 'green' ? '#2e7d32' : row.status === 'yellow' ? '#ed6c02' : '#d32f2f';
              return (
                <StyledTableRow key={row.name}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <FiberManualRecord
                        sx={{
                          fontSize: 10,
                          color: COLLECTION_STATUS_COLORS[row.status] || 'text.disabled',
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
                        color: badgeColor,
                        fontWeight: 600,
                        borderColor: badgeColor,
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
                        onSelectCollection(row);
                      }}
                    >
                      <InfoOutlined sx={{ fontSize: 18 }} />
                    </IconButton>
                  </TableCell>
                </StyledTableRow>
              );
            })
          )}
        </StyledTableBody>
      </Table>
    </StyledTableContainer>
  );
}

CollectionsTable.propTypes = {
  collections: PropTypes.array.isRequired,
  isLoading: PropTypes.bool.isRequired,
  onSelectCollection: PropTypes.func.isRequired,
};
