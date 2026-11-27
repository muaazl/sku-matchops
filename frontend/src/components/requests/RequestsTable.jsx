import React from 'react';
import PropTypes from 'prop-types';
import { Button, Chip, IconButton, Table, TableCell, TablePagination, TableRow } from '@mui/material';
import { InfoOutlined } from '@mui/icons-material';
import { HTTP_METHOD_COLORS } from '../../utils/constants';
import { fmtTime } from '../../utils/formatters';
import { EmptyState } from '../common/EmptyState';
import { HttpStatusChip } from '../common/HttpStatusChip';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function RequestsTable({
  rows,
  totalCount,
  isLoading,
  hasActiveFilters,
  page,
  rowsPerPage,
  onInspectRow,
  onClearFilters,
  onPageChange,
  onRowsPerPageChange,
}) {
  return (
    <StyledTableContainer>
      <Table aria-label="requests table">
        <StyledTableHead>
          <TableRow>
            <StyledHeaderCell width="10%">Method</StyledHeaderCell>
            <StyledHeaderCell width="18%">Path</StyledHeaderCell>
            <StyledHeaderCell width="18%">Client IP</StyledHeaderCell>
            <StyledHeaderCell width="15%">Status</StyledHeaderCell>
            <StyledHeaderCell width="15%">Duration</StyledHeaderCell>
            <StyledHeaderCell width="15%">Time</StyledHeaderCell>
            <StyledHeaderCell align="right">Actions</StyledHeaderCell>
          </TableRow>
        </StyledTableHead>
        <StyledTableBody>
          {isLoading ? (
            <TableSkeleton columns={7} rows={5} />
          ) : rows.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={7} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={InfoOutlined}
                  title="No API requests found"
                  description={
                    hasActiveFilters
                      ? 'No API requests matched your path query or status filter.'
                      : 'No incoming API requests have been recorded yet.'
                  }
                  action={
                    hasActiveFilters ? (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={onClearFilters}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Clear request filters
                      </Button>
                    ) : null
                  }
                  minHeight={240}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            rows.map((row) => (
              <StyledTableRow key={row.id} hover>
                <TableCell>
                  <Chip
                    size="small"
                    label={row.method}
                    color={HTTP_METHOD_COLORS[row.method] || 'default'}
                    variant="outlined"
                    sx={{ fontWeight: 'bold' }}
                  />
                </TableCell>
                <TableCell>{row.path}</TableCell>
                <TableCell>{row.ip_address || '127.0.0.1'}</TableCell>
                <TableCell>
                  <HttpStatusChip code={row.status_code} />
                </TableCell>
                <TableCell>{row.duration_ms} ms</TableCell>
                <TableCell>{fmtTime(row.created_at)}</TableCell>
                <TableCell align="right">
                  <IconButton size="small" onClick={() => onInspectRow(row)} aria-label="View request details">
                    <InfoOutlined sx={{ fontSize: 18 }} />
                  </IconButton>
                </TableCell>
              </StyledTableRow>
            ))
          )}
        </StyledTableBody>
      </Table>
      <TablePagination
        rowsPerPageOptions={[10, 25]}
        component="div"
        count={totalCount}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
      />
    </StyledTableContainer>
  );
}

RequestsTable.propTypes = {
  rows: PropTypes.array.isRequired,
  totalCount: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  hasActiveFilters: PropTypes.bool.isRequired,
  page: PropTypes.number.isRequired,
  rowsPerPage: PropTypes.number.isRequired,
  onInspectRow: PropTypes.func.isRequired,
  onClearFilters: PropTypes.func.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
