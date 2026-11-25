import React from 'react';
import PropTypes from 'prop-types';
import { Button, Table, TableCell, TablePagination, TableRow } from '@mui/material';
import { Label } from '@mui/icons-material';
import { formatGk } from '../../utils/formatters';
import { EmptyState } from '../common/EmptyState';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function ProcessedSkusTable({
  domain,
  skus,
  totalSkus,
  isLoading,
  hasActiveFilters,
  page,
  rowsPerPage,
  onSelectSku,
  onClearFilters,
  onPageChange,
  onRowsPerPageChange,
}) {
  const colSpan = 4;

  return (
    <StyledTableContainer>
      <Table aria-label="processed skus table">
        <StyledTableHead>
          <TableRow>
            <StyledHeaderCell>SKU Name</StyledHeaderCell>
            {domain === 'market' && <StyledHeaderCell width="15%">Categories</StyledHeaderCell>}
            <StyledHeaderCell width="45%">Generic Keywords</StyledHeaderCell>
            <StyledHeaderCell width="15%">Basic Type</StyledHeaderCell>
            {domain === 'food' && <StyledHeaderCell width="12%">Region</StyledHeaderCell>}
          </TableRow>
        </StyledTableHead>
        <StyledTableBody>
          {isLoading ? (
            <TableSkeleton columns={colSpan} rows={5} />
          ) : skus.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={colSpan} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={Label}
                  title="No matching SKUs found"
                  description={
                    hasActiveFilters
                      ? 'No SKUs in this job match your search or filter criteria. Try clearing or relaxing filters.'
                      : 'This job contains no processed SKU records.'
                  }
                  action={
                    hasActiveFilters ? (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={onClearFilters}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Clear SKU filters
                      </Button>
                    ) : null
                  }
                  minHeight={240}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            skus.map((row) => (
              <StyledTableRow key={row.id} hover onClick={() => onSelectSku(row)} sx={{ cursor: 'pointer' }}>
                <TableCell>{row.sku_name}</TableCell>
                {domain === 'market' && <TableCell>{row.categories || row.region}</TableCell>}
                <TableCell>{formatGk(row.generic_keywords || row.gk_json || row.gk)}</TableCell>
                <TableCell>{row.basic_type || row.bt}</TableCell>
                {domain === 'food' && <TableCell>{row.region}</TableCell>}
              </StyledTableRow>
            ))
          )}
        </StyledTableBody>
      </Table>

      <TablePagination
        rowsPerPageOptions={[100, 250, 500, 800]}
        component="div"
        count={totalSkus}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
      />
    </StyledTableContainer>
  );
}

ProcessedSkusTable.propTypes = {
  domain: PropTypes.string,
  skus: PropTypes.array.isRequired,
  totalSkus: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  hasActiveFilters: PropTypes.bool.isRequired,
  page: PropTypes.number.isRequired,
  rowsPerPage: PropTypes.number.isRequired,
  onSelectSku: PropTypes.func.isRequired,
  onClearFilters: PropTypes.func.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
