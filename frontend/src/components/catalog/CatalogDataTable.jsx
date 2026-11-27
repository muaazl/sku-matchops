import React from 'react';
import PropTypes from 'prop-types';
import { Button, Table, TableCell, TablePagination, TableRow, TableSortLabel } from '@mui/material';
import { Storage } from '@mui/icons-material';
import { EmptyState } from '../common/EmptyState';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function CatalogDataTable({
  columns,
  rows,
  total,
  isLoading,
  hasActiveFilters,
  sortModel,
  onHeaderClick,
  onClearAllFilters,
  page,
  pageSize,
  onPageChange,
  onRowsPerPageChange,
}) {
  return (
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
                      onClick={() => onHeaderClick(col)}
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
          ) : rows.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={columns.length} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={Storage}
                  title="No catalog records found"
                  description={
                    hasActiveFilters
                      ? 'No items matched your active search query or filter criteria. ' +
                        'Try clearing filters or using broader search terms.'
                      : 'There are no records in this dataset. Synchronize the local cache or switch domains.'
                  }
                  action={
                    hasActiveFilters ? (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={onClearAllFilters}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Clear search &amp; filters
                      </Button>
                    ) : null
                  }
                  minHeight={260}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            rows.map((row) => (
              <StyledTableRow key={row.id}>
                {columns.map((col) => {
                  const isGk = col.field === 'gk' || col.field === 'generic_keywords';
                  const isBt = col.field === 'bt' || col.field === 'basictype';
                  const fallback = isGk ? row.generic_keywords || row.gk : isBt ? row.basictype || row.bt : undefined;
                  const value = row[col.field] ?? fallback;
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
        count={total}
        rowsPerPage={pageSize}
        page={page}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
      />
    </StyledTableContainer>
  );
}

CatalogDataTable.propTypes = {
  columns: PropTypes.array.isRequired,
  rows: PropTypes.array.isRequired,
  total: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  hasActiveFilters: PropTypes.bool.isRequired,
  sortModel: PropTypes.array.isRequired,
  onHeaderClick: PropTypes.func.isRequired,
  onClearAllFilters: PropTypes.func.isRequired,
  page: PropTypes.number.isRequired,
  pageSize: PropTypes.number.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
