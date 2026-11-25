import React, { useEffect, useMemo, useState } from 'react';
import { Box, Button, MenuItem, TextField } from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { RequestInspectorDrawer } from '../components/requests/RequestInspectorDrawer';
import { RequestsTable } from '../components/requests/RequestsTable';
import { useApiRequests } from '../hooks/useApiRequests';
import { REQUEST_STATUS_OPTIONS } from '../utils/constants';

export default function Requests() {
  const [pathQuery, setPathQuery] = useState('');
  const [status, setStatus] = useState('all');
  const [selectedRow, setSelectedRow] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const {
    debouncedPathQuery,
    requests,
    isLoading,
    isFetching,
    refetch,
    selectedDetail,
    isDetailLoading,
    parsedHeaders,
    parsedQueryParams,
  } = useApiRequests({ pathQuery, status, selectedId: selectedRow?.id });

  useEffect(() => {
    setPage(0);
  }, [debouncedPathQuery, status]);

  const paginatedRows = useMemo(
    () => requests.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage),
    [requests, page, rowsPerPage]
  );

  return (
    <PageContainer>
      <PageHeader
        title="Requests"
        subtitle="Audit log of inbound API requests. Click a row to inspect payload & response."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <TextField
              size="small"
              placeholder="Filter by path…"
              value={pathQuery}
              onChange={(e) => setPathQuery(e.target.value)}
            />
            <TextField
              size="small"
              select
              label="Status"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              sx={{ width: 120 }}
            >
              {REQUEST_STATUS_OPTIONS.map((s) => (
                <MenuItem key={s} value={s}>
                  {s}
                </MenuItem>
              ))}
            </TextField>
            <Button
              variant="outlined"
              color="inherit"
              size="small"
              aria-label="Refresh Requests"
              onClick={() => refetch()}
              disabled={isLoading || isFetching}
              startIcon={
                <Refresh
                  sx={{
                    fontSize: 16,
                    ...(isFetching && {
                      '@keyframes spin': {
                        from: { transform: 'rotate(0deg)' },
                        to: { transform: 'rotate(360deg)' },
                      },
                      animation: 'spin 0.9s linear infinite',
                    }),
                  }}
                />
              }
              sx={{ height: 40, px: 1.75, flexShrink: 0 }}
            >
              Refresh
            </Button>
          </Box>
        }
      />

      <RequestsTable
        rows={paginatedRows}
        totalCount={requests.length}
        isLoading={isLoading}
        hasActiveFilters={Boolean(pathQuery || status !== 'all')}
        page={page}
        rowsPerPage={rowsPerPage}
        onInspectRow={(row) => {
          setSelectedRow(row);
          setActiveTab(0);
        }}
        onClearFilters={() => {
          setPathQuery('');
          setStatus('all');
          setPage(0);
        }}
        onPageChange={(event, newPage) => setPage(newPage)}
        onRowsPerPageChange={(event) => {
          setRowsPerPage(parseInt(event.target.value, 10));
          setPage(0);
        }}
      />

      <RequestInspectorDrawer
        selectedRow={selectedRow}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onClose={() => setSelectedRow(null)}
        selectedDetail={selectedDetail}
        isDetailLoading={isDetailLoading}
        parsedHeaders={parsedHeaders}
        parsedQueryParams={parsedQueryParams}
      />
    </PageContainer>
  );
}
