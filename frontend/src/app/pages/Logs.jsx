import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Button,
  TextField,
  MenuItem,
  Typography,
  Stack,
  FormControlLabel,
  Switch,
  CircularProgress,
} from '@mui/material';
import { Download, RefreshCw, Search, Terminal as TerminalIcon } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader } from '../components/ui';
import { getLogs } from '../api';
import { download } from '../utils';

export default function Logs() {
  const { enqueueSnackbar } = useSnackbar();
  const [lines, setLines] = useState(500);
  const [logsText, setLogsText] = useState('');
  const [filterQuery, setFilterQuery] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  const terminalRef = useRef(null);

  const fetchLogs = async (silent = false) => {
    if (!silent) setIsLoading(true);
    try {
      const data = await getLogs(lines);
      setLogsText(data.logs || '');
    } catch (err) {
      if (!silent) {
        enqueueSnackbar(`Failed to fetch logs: ${err.message}`, { variant: 'error' });
      }
    } finally {
      if (!silent) setIsLoading(false);
    }
  };

  // Initial load and when lines count changes
  useEffect(() => {
    fetchLogs();
  }, [lines]);

  // Auto-refresh interval
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchLogs(true);
    }, 3000);

    return () => clearInterval(interval);
  }, [autoRefresh, lines]);

  // Auto-scroll logic
  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logsText, autoScroll]);

  const handleDownload = () => {
    try {
      download(`sku-matchops_${new Date().toISOString().slice(0, 10)}_logs.txt`, logsText, 'text/plain;charset=utf-8');
      enqueueSnackbar('Logs downloaded successfully', { variant: 'success' });
    } catch (e) {
      enqueueSnackbar('Failed to download logs', { variant: 'error' });
    }
  };

  const parsedLines = useMemo(() => {
    if (!logsText) return [];

    const rawLines = logsText.split('\n');
    const filtered = rawLines.filter((line) => {
      if (!filterQuery) return true;
      return line.toLowerCase().includes(filterQuery.toLowerCase());
    });

    return filtered.map((line, idx) => {
      let color = '#d4d4d4'; // default light gray
      let fontWeight = 'normal';

      if (line.includes(' - ERROR - ') || line.includes(' - CRITICAL - ')) {
        color = '#f87171'; // red
        fontWeight = 'bold';
      } else if (line.includes(' - WARNING - ')) {
        color = '#fbbf24'; // orange/yellow
        fontWeight = 'bold';
      } else if (line.includes(' - INFO - ')) {
        color = '#34d399'; // green-ish
      } else if (line.includes(' - DEBUG - ')) {
        color = '#9ca3af'; // gray
      }

      return { text: line, color, fontWeight, id: idx };
    });
  }, [logsText, filterQuery]);

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
              startIcon={<Download size={15} />}
              onClick={handleDownload}
              disabled={!logsText}
            >
              Download
            </Button>
            <Button
              variant="contained"
              startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <RefreshCw size={15} />}
              onClick={() => fetchLogs()}
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
                    <Search size={16} />
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

      <Card sx={{ bgcolor: '#0b0f17', border: '1px solid #1e293b' }}>
        <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: '1px solid #1e293b',
              pb: 1,
              mb: 1.5,
              px: 1,
            }}
          >
            <Stack direction="row" spacing={1} alignItems="center">
              <TerminalIcon size={16} color="#94a3b8" />
              <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#94a3b8', fontWeight: 600 }}>
                console.log
              </Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Showing {parsedLines.length} of {logsText.split('\n').filter(Boolean).length} lines
            </Typography>
          </Box>
          <Box
            ref={terminalRef}
            sx={{
              maxHeight: '600px',
              minHeight: '350px',
              overflowY: 'auto',
              fontFamily: "'Fira Code', 'Courier New', Courier, monospace",
              fontSize: '0.8rem',
              lineHeight: 1.6,
              color: '#d4d4d4',
              backgroundColor: '#0c0f16',
              p: 2,
              borderRadius: 1,
              '&::-webkit-scrollbar': {
                width: '8px',
                height: '8px',
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: '#0c0f16',
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: '#1e293b',
                borderRadius: '4px',
              },
            }}
          >
            {parsedLines.length === 0 ? (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: '350px',
                  color: 'text.secondary',
                }}
              >
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {isLoading ? 'Loading logs...' : 'No logs found.'}
                </Typography>
              </Box>
            ) : (
              parsedLines.map((line) => (
                <div
                  key={line.id}
                  style={{
                    color: line.color,
                    fontWeight: line.fontWeight,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                    paddingBottom: '2px',
                  }}
                >
                  {line.text}
                </div>
              ))
            )}
          </Box>
        </CardContent>
      </Card>
    </PageContainer>
  );
}
