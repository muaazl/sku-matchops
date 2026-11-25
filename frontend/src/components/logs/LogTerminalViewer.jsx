import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, CircularProgress, Stack, Typography } from '@mui/material';
import { Terminal as TerminalIcon } from '@mui/icons-material';

export function LogTerminalViewer({ terminalRef, parsedLines, totalNonEmptyLines, isLoading, filterQuery }) {
  return (
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
            <TerminalIcon sx={{ fontSize: 16, color: '#94a3b8' }} />
            <Typography variant="caption" sx={{ fontFamily: 'monospace', color: '#94a3b8', fontWeight: 600 }}>
              console.log
            </Typography>
          </Stack>
          <Typography variant="caption" color="text.secondary">
            Showing {parsedLines.length} of {totalNonEmptyLines} lines
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
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '350px',
                color: 'text.secondary',
                gap: 1.5,
              }}
            >
              {isLoading ? (
                <>
                  <CircularProgress size={28} />
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    Streaming logs...
                  </Typography>
                </>
              ) : (
                <>
                  <TerminalIcon sx={{ fontSize: 32, opacity: 0.6 }} />
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {filterQuery ? 'No log lines matched your filter query.' : 'No log lines available.'}
                  </Typography>
                </>
              )}
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
  );
}

LogTerminalViewer.propTypes = {
  terminalRef: PropTypes.oneOfType([PropTypes.func, PropTypes.shape({ current: PropTypes.any })]),
  parsedLines: PropTypes.array.isRequired,
  totalNonEmptyLines: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  filterQuery: PropTypes.string.isRequired,
};
