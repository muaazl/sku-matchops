import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Chip,
  CircularProgress,
  Divider,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Tabs,
  Typography,
} from '@mui/material';
import { HTTP_METHOD_COLORS } from '../../utils/constants';
import { fmtTime } from '../../utils/formatters';
import { HttpStatusChip } from '../common/HttpStatusChip';
import { JsonBlock } from '../common/JsonBlock';
import { KeyValueTable } from '../common/KeyValueTable';
import { SideDrawer } from '../common/SideDrawer';
import { StyledTableContainer } from '../common/StyledTable';

export function RequestInspectorDrawer({
  selectedRow,
  activeTab,
  onTabChange,
  onClose,
  selectedDetail,
  isDetailLoading,
  parsedHeaders,
  parsedQueryParams,
}) {
  return (
    <SideDrawer open={!!selectedRow} onClose={onClose} title="Request Inspector" width={600}>
      {selectedRow && (
        <Box>
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', mb: 1, flexWrap: 'wrap' }}>
            <Chip
              size="small"
              label={selectedRow.method}
              color={HTTP_METHOD_COLORS[selectedRow.method] || 'default'}
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

          <Tabs
            value={activeTab}
            onChange={(e, val) => onTabChange(val)}
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
                      <TableCell sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>{selectedRow.path}</TableCell>
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
                <KeyValueTable data={parsedHeaders} emptyMessage="No headers captured." />
              ))}

            {activeTab === 2 &&
              (isDetailLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                <KeyValueTable data={parsedQueryParams} emptyMessage="No query parameters." />
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
  );
}

RequestInspectorDrawer.propTypes = {
  selectedRow: PropTypes.object,
  activeTab: PropTypes.number.isRequired,
  onTabChange: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
  selectedDetail: PropTypes.object,
  isDetailLoading: PropTypes.bool.isRequired,
  parsedHeaders: PropTypes.object.isRequired,
  parsedQueryParams: PropTypes.object.isRequired,
};
