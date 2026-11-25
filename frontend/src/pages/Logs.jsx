import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CircularProgress,
  FormControlLabel,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import { Download, Refresh, Search } from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { LogTerminalViewer } from '../components/logs/LogTerminalViewer';
import { useLogs } from '../hooks/useLogs';
import { download } from '../utils/csvUtils';

export default function Logs() {
  const { enqueueSnackbar } = useSnackbar();
  const [lines, setLines] = useState(500);
  const [filterQuery, setFilterQuery] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const terminalRef = useRef(null);

  const { logsText, parsedLines, totalNonEmptyLines, isLoading, refreshLogs } = useLogs({
    lines,
    filterQuery,
    autoRefresh,
  });

  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logsText, autoScroll]);

  const handleDownload = () => {
    try {
      download(`sku-matchops_${new Date().toISOString().slice(0, 10)}_logs.txt`, logsText, 'text/plain;charset=utf-8');
      enqueueSnackbar('Logs downloaded successfully', { variant: 'success' });
    } catch {
      enqueueSnackbar('Failed to download logs', { variant: 'error' });
    }
  };

  return (
    <PageContainer>
      <PageHeader
        title="System Logs"
        subtitle="Real-time log viewer for the server, vector searches, and rules engine executions."
        actions={
          <Stack direction="row" spacing={1.5}>
            <Button
              variant="outlined"
              color="inherit"
              startIcon={<Download sx={{ fontSize: 15 }} />}
              onClick={handleDownload}
              disabled={!logsText}
            >
              Download
            </Button>
            <Button
              variant="contained"
              startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <Refresh sx={{ fontSize: 15 }} />}
              onClick={refreshLogs}
              disabled={isLoading}
            >
              Refresh
            </Button>
          </Stack>
        }
      />

      <Card sx={{ p: 2, mb: 3 }}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ md: 'center' }}>
          <TextField
            select
            size="small"
            label="Log Limit (Lines)"
            value={lines}
            onChange={(e) => setLines(Number(e.target.value))}
            sx={{ width: 180 }}
          >
            <MenuItem value={100}>Last 100 lines</MenuItem>
            <MenuItem value={200}>Last 200 lines</MenuItem>
            <MenuItem value={500}>Last 500 lines</MenuItem>
            <MenuItem value={1000}>Last 1000 lines</MenuItem>
            <MenuItem value={2000}>Last 2000 lines</MenuItem>
          </TextField>

          <TextField
            size="small"
            label="Filter logs..."
            variant="outlined"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            slotProps={{
              input: {
                startAdornment: (
                  <Box sx={{ color: 'text.secondary', mr: 1, display: 'flex', alignItems: 'center' }}>
                    <Search sx={{ fontSize: 16 }} />
                  </Box>
                ),
              },
            }}
            sx={{ flex: 1 }}
          />

          <FormControlLabel
            control={<Switch checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} size="small" />}
            label={
              <Typography variant="body2" color="text.secondary">
                Auto-refresh (3s)
              </Typography>
            }
          />

          <FormControlLabel
            control={<Switch checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} size="small" />}
            label={
              <Typography variant="body2" color="text.secondary">
                Auto-scroll
              </Typography>
            }
          />
        </Stack>
      </Card>

      <LogTerminalViewer
        terminalRef={terminalRef}
        parsedLines={parsedLines}
        totalNonEmptyLines={totalNonEmptyLines}
        isLoading={isLoading}
        filterQuery={filterQuery}
      />
    </PageContainer>
  );
}
