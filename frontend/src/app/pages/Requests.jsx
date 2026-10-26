import React, { useState } from 'react';
import {
  Box,
  IconButton,
  TextField,
  MenuItem,
  Typography,
  Divider,
  Chip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableRow,
  CircularProgress,
  Tooltip,
  TablePagination,
} from '@mui/material';
import { RefreshCw, Info } from 'lucide-react';
import { PageContainer, PageHeader, HttpStatusChip, SideDrawer } from '../components/ui';
import { JsonBlock } from '../components/JsonBlock';
import { fmtTime } from '../utils';
import { useQuery } from '@tanstack/react-query';

import { getApiRequests, getApiRequest } from '../api';
import { useDebounce } from '../hooks/useDebounce';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

const STATUS_OPTS = ['all', '200', '400', '401', '500'];
const METHOD_COLOR = { GET: 'info', POST: 'success', PUT: 'warning', DELETE: 'error' };

export default function Requests() {
  const [pathQuery, setPathQuery] = useState('');
  const [status, setStatus] = useState('all');
  const [selectedRow, setSelectedRow] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const debouncedPathQuery = useDebounce(pathQuery, 350);

  // Reset page when filter changes
  React.useEffect(() => {
    setPage(0);
  }, [debouncedPathQuery, status]);

  // Parse status filter for server-side
  const statusCodeParam = status === 'all' ? undefined : parseInt(status, 10);

  // Fetch requests list
  const {
    data: serverRequests = [],
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['api-requests', debouncedPathQuery, status],
    queryFn: () =>
      getApiRequests({
        path: debouncedPathQuery || undefined,
        status_code: statusCodeParam,
      }),
    refetchInterval: 15000, // Poll every 15 seconds for live updates
    refetchIntervalInBackground: false, // Don't poll when tab is not active
  });

  // Fetch selected request details
  const selectedId = selectedRow?.id;
  const { data: selectedDetail, isLoading: isDetailLoading } = useQuery({
    queryKey: ['api-request', selectedId],
    queryFn: () => getApiRequest(selectedId),
    enabled: !!selectedId,
  });

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const paginatedRows = React.useMemo(() => {
    return serverRequests.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [serverRequests, page, rowsPerPage]);

  const parsedHeaders = React.useMemo(() => {
    if (!selectedDetail?.headers_json) return {};
    if (typeof selectedDetail.headers_json === 'object') return selectedDetail.headers_json;
    try {
      return JSON.parse(selectedDetail.headers_json);
    } catch (e) {
      console.error('Failed to parse headers_json', e);
      return {};
    }
  }, [selectedDetail?.headers_json]);

  const parsedQueryParams = React.useMemo(() => {
    if (!selectedDetail?.query_params_json) return {};
    if (typeof selectedDetail.query_params_json === 'object') return selectedDetail.query_params_json;
    try {
      return JSON.parse(selectedDetail.query_params_json);
    } catch (e) {
      console.error('Failed to parse query_params_json', e);
      return {};
    }
  }, [selectedDetail?.query_params_json]);


  const renderHeaders = () => {
    const headerEntries = Object.entries(parsedHeaders);
    if (headerEntries.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
          No headers captured.
        </Typography>
      );
    }

    return (
      <StyledTableContainer>
        <Table size="small">
          <TableBody>
            {headerEntries.map(([key, val]) => (
              <TableRow key={key}>
                <TableCell
                  sx={{
                    fontWeight: 'bold',
                    fontFamily: 'monospace',
                    fontSize: 11,
                    width: '35%',
                    wordBreak: 'break-all',
                    py: 1,
                  }}
                >
                  {key}
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, wordBreak: 'break-all', py: 1 }}>
                  {String(val)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </StyledTableContainer>
    );
  };

  const renderQueryParams = () => {
    const queryEntries = Object.entries(parsedQueryParams);
    if (queryEntries.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
          No query parameters.
        </Typography>
      );
    }

    return (
      <StyledTableContainer>
        <Table size="small">
          <TableBody>
            {queryEntries.map(([key, val]) => (
              <TableRow key={key}>
                <TableCell
                  sx={{
                    fontWeight: 'bold',
                    fontFamily: 'monospace',
                    fontSize: 11,
                    width: '35%',
                    wordBreak: 'break-all',
                    py: 1,
                  }}
                >
                  {key}
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, wordBreak: 'break-all', py: 1 }}>
                  {String(val)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </StyledTableContainer>
    );
  };

  return (
    <PageContainer>
      <PageHeader
        title="Requests"
        subtitle="Audit log of inbound API requests. Click a row to inspect payload & response."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
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
              {STATUS_OPTS.map((s) => (
                <MenuItem key={s} value={s}>
                  {s}
                </MenuItem>
              ))}
            </TextField>
            <Tooltip title="Refresh Logs">
              <IconButton onClick={() => refetch()} disabled={isLoading || isFetching} aria-label="Refresh requests">
                <RefreshCw size={18} className={isFetching ? 'animate-spin' : ''} />
              </IconButton>
            </Tooltip>
          </Box>
        }
      />
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
            ) : paginatedRows.length === 0 ? (
              <StyledTableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No requests found.
                  </Typography>
                </TableCell>
              </StyledTableRow>
            ) : (
              paginatedRows.map((row) => (
                <StyledTableRow key={row.id} hover>
                  <TableCell>
                    <Chip
                      size="small"
                      label={row.method}
                      color={METHOD_COLOR[row.method] || 'default'}
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
                    <IconButton
                      size="small"
                      onClick={() => {
                        setSelectedRow(row);
                        setActiveTab(0);
                      }}
                      aria-label="View request details"
                    >
                      <Info size={18} />
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
          count={serverRequests.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
        />
      </StyledTableContainer>

      <SideDrawer open={!!selectedRow} onClose={() => setSelectedRow(null)} title="Request Inspector" width={600}>
        {selectedRow && (
          <Box>
            {/* HTTP Path & Status code overview */}
            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mb: 1, flexWrap: 'wrap' }}>
              <Chip
                size="small"
                label={selectedRow.method}
                color={METHOD_COLOR[selectedRow.method] || 'default'}
                variant="outlined"
                sx={{ fontWeight: 'bold' }}
              />
              <code
                style={{
                  fontSize: 13,
                  background: 'rgba(0, 0, 0, 0.05)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  wordBreak: 'break-all',
                }}
              >
                {selectedRow.path}
              </code>
              <HttpStatusChip code={selectedRow.status_code} />
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ mb: 2 }}>
              ID: <code>{selectedRow.id}</code> · {fmtTime(selectedRow.created_at)}
            </Typography>

            <Divider sx={{ mb: 2 }} />

            {/* Tabs */}
            <Tabs
              value={activeTab}
              onChange={(e, val) => setActiveTab(val)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
            >
              <Tab label="General" sx={{ textTransform: 'none', fontWeight: 'bold' }} />
              <Tab label="Headers" sx={{ textTransform: 'none', fontWeight: 'bold' }} />
              <Tab label="Query Params" sx={{ textTransform: 'none', fontWeight: 'bold' }} />
              <Tab label="Payload (Body)" sx={{ textTransform: 'none', fontWeight: 'bold' }} />
              <Tab label="Response" sx={{ textTransform: 'none', fontWeight: 'bold' }} />
            </Tabs>

            {/* Tab Panels */}
            <Box>
              {activeTab === 0 && (
                <StyledTableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold', width: '35%' }}>Method</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace' }}>{selectedRow.method}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Path</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                          {selectedRow.path}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Status</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <HttpStatusChip code={selectedRow.status_code} />
                            <Typography variant="body2" component="span" color="text.secondary">
                              {selectedRow.status_code >= 200 && selectedRow.status_code < 300 ? 'Success' : 'Error'}
                            </Typography>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Client IP</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace' }}>{selectedRow.ip_address || '127.0.0.1'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Duration</TableCell>
                        <TableCell>{selectedRow.duration_ms} ms</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Time</TableCell>
                        <TableCell>{fmtTime(selectedRow.created_at)}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </StyledTableContainer>
              )}

              {activeTab === 1 &&
                (isDetailLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : (
                  renderHeaders()
                ))}

              {activeTab === 2 &&
                (isDetailLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : (
                  renderQueryParams()
                ))}

              {activeTab === 3 &&
                (isDetailLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : !selectedDetail?.payload_json_redacted ? (
                  <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                    No payload recorded.
                  </Typography>
                ) : (
                  <JsonBlock value={selectedDetail.payload_json_redacted} />
                ))}

              {activeTab === 4 &&
                (isDetailLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : !selectedDetail?.response_json ? (
                  <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                    No response recorded.
                  </Typography>
                ) : (
                  <JsonBlock value={selectedDetail.response_json} />
                ))}
            </Box>
          </Box>
        )}
      </SideDrawer>
    </PageContainer>
  );
}
