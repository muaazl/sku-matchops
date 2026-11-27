import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, IconButton, Stack, Table, TableCell, TablePagination, TableRow, Tooltip } from '@mui/material';
import { Add, Balance, DeleteOutline, DragIndicator, Edit } from '@mui/icons-material';
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

export function RulesTable({
  rows,
  totalCount,
  isLoading,
  domain,
  canEditRules,
  lockedHint,
  page,
  rowsPerPage,
  draggedIndex,
  onDragStart,
  onDragOver,
  onDragEnd,
  onEditRule,
  onDeleteRule,
  onNewRule,
  onPageChange,
  onRowsPerPageChange,
}) {
  return (
    <StyledTableContainer>
      <Table aria-label="rules table">
        <StyledTableHead>
          <TableRow>
            <StyledHeaderCell width="50px" />
            <StyledHeaderCell width="15%">Rule ID</StyledHeaderCell>
            <StyledHeaderCell width="12%">Domain</StyledHeaderCell>
            <StyledHeaderCell>Description</StyledHeaderCell>
            <StyledHeaderCell width="12%">Status</StyledHeaderCell>
            <StyledHeaderCell align="right" width="10%">
              Actions
            </StyledHeaderCell>
          </TableRow>
        </StyledTableHead>
        <StyledTableBody>
          {isLoading ? (
            <TableSkeleton columns={6} rows={5} />
          ) : rows.length === 0 ? (
            <StyledTableRow>
              <TableCell colSpan={6} sx={{ p: 0, border: 'none' }}>
                <EmptyState
                  icon={Balance}
                  title="No rules found"
                  description={
                    domain !== 'all'
                      ? `No rules found for the "${domain}" domain. Try selecting "All Domain".`
                      : 'Define automated business rules to apply specific tags, categories, ' +
                        'or actions to incoming SKUs.'
                  }
                  action={
                    canEditRules && domain === 'all' ? (
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<Add sx={{ fontSize: 16 }} />}
                        onClick={onNewRule}
                        sx={{ textTransform: 'none', fontWeight: 600 }}
                      >
                        New rule
                      </Button>
                    ) : null
                  }
                  minHeight={240}
                />
              </TableCell>
            </StyledTableRow>
          ) : (
            rows.map((row, index) => {
              const isItemDragged = draggedIndex === page * rowsPerPage + index;
              return (
                <StyledTableRow
                  key={row.rule_id}
                  hover
                  onDragOver={(e) => onDragOver(e, index)}
                  sx={{
                    cursor: 'default',
                    opacity: isItemDragged ? 0.4 : 1,
                    backgroundColor: isItemDragged ? 'action.hover' : 'inherit',
                    transition: 'opacity 0.2s ease, background-color 0.2s ease',
                  }}
                >
                  <TableCell onClick={(e) => e.stopPropagation()} sx={{ width: '50px' }}>
                    {domain !== 'all' && canEditRules ? (
                      <Box
                        draggable
                        onDragStart={(e) => onDragStart(e, index)}
                        onDragEnd={onDragEnd}
                        sx={{
                          cursor: 'grab',
                          '&:active': { cursor: 'grabbing' },
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          p: 0.5,
                          color: 'text.secondary',
                        }}
                        title="Drag to reorder"
                      >
                        <DragIndicator sx={{ fontSize: 16 }} />
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          p: 0.5,
                          color: 'text.disabled',
                          opacity: 0.5,
                        }}
                        title={canEditRules ? 'Select a specific domain to enable reordering' : lockedHint}
                      >
                        <DragIndicator sx={{ fontSize: 16 }} />
                      </Box>
                    )}
                  </TableCell>
                  <TableCell>{row.rule_id}</TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
                  <TableCell>{row.description}</TableCell>
                  <TableCell>
                    <StatusChip status={row.is_active ? 'active' : 'disabled'} />
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          onEditRule(row);
                        }}
                        aria-label="Edit rule"
                        title="Edit rule"
                      >
                        <Edit sx={{ fontSize: 18 }} />
                      </IconButton>
                      <Tooltip title={lockedHint || 'Delete rule'}>
                        <span>
                          <IconButton
                            size="small"
                            color="error"
                            disabled={!canEditRules}
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteRule(row.rule_id);
                            }}
                            aria-label="Delete rule"
                          >
                            <DeleteOutline sx={{ fontSize: 18 }} />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  </TableCell>
                </StyledTableRow>
              );
            })
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

RulesTable.propTypes = {
  rows: PropTypes.array.isRequired,
  totalCount: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  domain: PropTypes.string.isRequired,
  canEditRules: PropTypes.bool.isRequired,
  lockedHint: PropTypes.string.isRequired,
  page: PropTypes.number.isRequired,
  rowsPerPage: PropTypes.number.isRequired,
  draggedIndex: PropTypes.number,
  onDragStart: PropTypes.func.isRequired,
  onDragOver: PropTypes.func.isRequired,
  onDragEnd: PropTypes.func.isRequired,
  onEditRule: PropTypes.func.isRequired,
  onDeleteRule: PropTypes.func.isRequired,
  onNewRule: PropTypes.func.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onRowsPerPageChange: PropTypes.func.isRequired,
};
