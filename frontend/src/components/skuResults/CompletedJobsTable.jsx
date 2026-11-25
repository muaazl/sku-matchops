import React from 'react';
import PropTypes from 'prop-types';
import { Button, IconButton, Table, TableCell, TablePagination, TableRow } from '@mui/material';
import { InfoOutlined, TableChart } from '@mui/icons-material';
import { EmptyState } from '../common/EmptyState';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function CompletedJobsTable({
  jobs,
  totalJobs,
  isLoading,
  hasActiveFilters,
  page,
  rowsPerPage,
  onSelectJob,
  onResetFilters,
  onNavigateProcess,
  onPageChange,
  onRowsPerPageChange,
}) {
  return (
    <StyledTableContainer>
      <Table aria-label="jobs table">
        <StyledTableHead>
          <TableRow>
            <StyledHeaderCell width="15%">Job ID</StyledHeaderCell>
            <StyledHeaderCell width="25%">Sheet Name</StyledHeaderCell>
            <StyledHeaderCell width="15%">Domain</StyledHeaderCell>
            <StyledHeaderCell width="15%">Task</StyledHeaderCell>
            <StyledHeaderCell align="right" width="10%">
              Total SKUs
            </StyledHeaderCell>
            <StyledHeaderCell align="right" width="10%">
              High Conf
            </StyledHeaderCell>
            <StyledHeaderCell align="right" width="10%">
              Details
            </StyledHeaderCell>
          </TableRow>
        </StyledTableHead>
        <StyledTableBody>
          {isLoading ? (
            <TableSkeleton columns={7} rows={5} />
          ) : jobs.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={7} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={TableChart}
                  title="No completed jobs found"
                  description={
                    hasActiveFilters
                      ? 'No completed SKU matching jobs match your active filters.'
                      : 'There are no completed batch jobs available to review yet.'
                  }
                  action={
                    hasActiveFilters ? (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={onResetFilters}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Reset filters
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        size="small"
                        onClick={onNavigateProcess}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Process new SKUs
                      </Button>
                    )
                  }
                  minHeight={240}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            jobs.map((row) => (
              <StyledTableRow key={row.id} hover onClick={() => onSelectJob(row)} sx={{ cursor: 'pointer' }}>
                <TableCell>{row.id}</TableCell>
                <TableCell>{row.target_sheet || row.sheet_name || 'N/A'}</TableCell>
                <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
                <TableCell sx={{ textTransform: 'capitalize' }}>{row.task || row.type}</TableCell>
                <TableCell align="right">{row.total_items ?? '-'}</TableCell>
                <TableCell align="right">{row.high_conf ?? '-'}</TableCell>
                <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                  <IconButton
                    size="small"
                    aria-label="View job details"
                    sx={{ color: 'text.secondary' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectJob(row);
                    }}
                  >
                    <InfoOutlined sx={{ fontSize: 18 }} />
                  </IconButton>
                </TableCell>
              </StyledTableRow>
            ))
          )}
        </StyledTableBody>
      </Table>
      <TablePagination
        rowsPerPageOptions={[10, 25, 50]}
        component="div"
        count={totalJobs}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
      />
    </StyledTableContainer>
  );
}

CompletedJobsTable.propTypes = {
  jobs: PropTypes.array.isRequired,
  totalJobs: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  hasActiveFilters: PropTypes.bool.isRequired,
  page: PropTypes.number.isRequired,
  rowsPerPage: PropTypes.number.isRequired,
  onSelectJob: PropTypes.func.isRequired,
  onResetFilters: PropTypes.func.isRequired,
  onNavigateProcess: PropTypes.func.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
