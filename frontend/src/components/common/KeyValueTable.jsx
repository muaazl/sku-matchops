import React from 'react';
import PropTypes from 'prop-types';
import { Table, TableBody, TableCell, TableRow, Typography } from '@mui/material';
import { StyledTableContainer } from './StyledTable';

export function KeyValueTable({ data, emptyMessage }) {
  const entries = Object.entries(data || {});
  if (entries.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
        {emptyMessage}
      </Typography>
    );
  }

  return (
    <StyledTableContainer>
      <Table size="small">
        <TableBody>
          {entries.map(([key, val]) => (
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
}

KeyValueTable.propTypes = {
  data: PropTypes.object,
  emptyMessage: PropTypes.string.isRequired,
};
