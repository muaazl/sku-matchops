import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Chip, IconButton, Table, TableCell, TablePagination, TableRow, Tooltip } from '@mui/material';
import { Cancel, PendingActions, PlayArrow, Replay } from '@mui/icons-material';
import { fmtDuration } from '../../utils/formatters';
import { EmptyState } from '../common/EmptyState';
import { StatusChip } from '../common/StatusChip';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../common/StyledTable';
import { TableSkeleton } from '../common/TableSkeleton';

export function JobsTable({
  jobs,
  totalJobs,
  isLoading,
  filter,
  page,
  rowsPerPage,
  isCancelling,
  isRetrying,
  onRowClick,
  onCancelClick,
  onRetryClick,
  onResetFilter,
  onNavigateProcess,
  onPageChange,
  onRowsPerPageChange,
}) {
  return (
    <StyledTableContainer>
      <Table aria-label="jobs table">
        <StyledTableHead>
          <TableRow>
            <StyledHeaderCell width="12%">Job ID</StyledHeaderCell>
            <StyledHeaderCell width="25%">Sheet Name</StyledHeaderCell>
            <StyledHeaderCell width="15%">Domain</StyledHeaderCell>
            <StyledHeaderCell width="15%">Task</StyledHeaderCell>
            <StyledHeaderCell width="15%">Status</StyledHeaderCell>
            <StyledHeaderCell align="right">Total SKUs</StyledHeaderCell>
            <StyledHeaderCell align="right">High Conf</StyledHeaderCell>
            <StyledHeaderCell align="right">Duration</StyledHeaderCell>
            <StyledHeaderCell align="right" width="10%">
              Actions
            </StyledHeaderCell>
          </TableRow>
        </StyledTableHead>
        <StyledTableBody>
          {isLoading ? (
            <TableSkeleton columns={9} rows={5} />
          ) : jobs.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={9} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={PendingActions}
                  title={filter === 'all' ? 'No jobs found' : `No ${filter} jobs`}
                  description={
                    filter === 'all'
                      ? 'There are no batch SKU processing jobs recorded yet. Upload a batch to get started.'
                      : `There are currently no jobs with status "${filter}". Try selecting a different filter.`
                  }
                  action={
                    filter === 'all' ? (
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<PlayArrow sx={{ fontSize: 16 }} />}
                        onClick={onNavigateProcess}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Process new SKUs
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={onResetFilter}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        Show all jobs
                      </Button>
                    )
                  }
                  minHeight={260}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            jobs.map((row) => {
              const isUpload = row.task === 'upload' || row.type === 'upload';
              const canCancel = ['running', 'queued'].includes(row.status) && !isUpload;
              return (
                <StyledTableRow
                  key={row.id}
                  hover
                  onClick={() => onRowClick(row, isUpload)}
                  sx={{ cursor: 'pointer', transition: 'background-color 150ms ease' }}
                >
                  <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600, fontSize: '0.85rem' }}>{row.id}</TableCell>
                  <TableCell
                    sx={{
                      fontWeight: 500,
                      maxWidth: 220,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    <Tooltip title={row.target_sheet || row.sheet_name || 'N/A'}>
                      <span>{row.target_sheet || row.sheet_name || 'N/A'}</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>
                    <Chip
                      size="small"
                      label={row.domain || 'All'}
                      variant="outlined"
                      sx={{ fontWeight: 500, height: 22, fontSize: '0.75rem' }}
                    />
                  </TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>{row.task || row.type}</TableCell>
                  <TableCell>
                    <StatusChip status={row.status} />
                  </TableCell>
                  <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>
                    {row.total_items?.toLocaleString() ?? '-'}
                  </TableCell>
                  <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>
                    {row.high_conf?.toLocaleString() ?? '-'}
                  </TableCell>
                  <TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>
                    {fmtDuration(row.duration_minutes)}
                  </TableCell>
                  <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5 }}>
                      <Tooltip title={canCancel ? 'Cancel job' : 'Nothing to cancel'}>
                        <span>
                          <IconButton
                            aria-label={canCancel ? 'Cancel job' : 'Nothing to cancel'}
                            size="small"
                            disabled={!canCancel || isCancelling}
                            onClick={() => onCancelClick(row.id)}
                          >
                            <Cancel sx={{ fontSize: 17 }} />
                          </IconButton>
                        </span>
                      </Tooltip>
                      {!isUpload && (
                        <Tooltip title="Retry job">
                          <IconButton
                            aria-label="Retry job"
                            size="small"
                            disabled={isRetrying}
                            onClick={() => onRetryClick(row.id)}
                          >
                            <Replay sx={{ fontSize: 16 }} />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  </TableCell>
                </StyledTableRow>
              );
            })
          )}
        </StyledTableBody>
      </Table>
      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
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

JobsTable.propTypes = {
  jobs: PropTypes.array.isRequired,
  totalJobs: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  filter: PropTypes.string.isRequired,
  page: PropTypes.number.isRequired,
  rowsPerPage: PropTypes.number.isRequired,
  isCancelling: PropTypes.bool.isRequired,
  isRetrying: PropTypes.bool.isRequired,
  onRowClick: PropTypes.func.isRequired,
  onCancelClick: PropTypes.func.isRequired,
  onRetryClick: PropTypes.func.isRequired,
  onResetFilter: PropTypes.func.isRequired,
  onNavigateProcess: PropTypes.func.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
