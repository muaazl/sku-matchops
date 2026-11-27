import React from 'react';
import PropTypes from 'prop-types';
import { Box, Skeleton, Stack, TableCell } from '@mui/material';
import { StyledTableRow } from './StyledTable';

export function TableSkeleton({ columns, rows = 5, height = 44, sx = {} }) {
  if (typeof columns === 'number' && columns > 0) {
    return (
      <>
        {[...Array(rows)].map((_, rowIndex) => (
          <StyledTableRow key={rowIndex}>
            {[...Array(columns)].map((__, colIndex) => (
              <TableCell key={colIndex}>
                <Skeleton variant="text" width={colIndex === 0 ? '40%' : '80%'} height={24} />
              </TableCell>
            ))}
          </StyledTableRow>
        ))}
      </>
    );
  }

  return (
    <Box sx={{ width: '100%', overflow: 'hidden', ...sx }}>
      <Stack spacing={1.5}>
        <Skeleton variant="rounded" height={40} sx={{ borderRadius: 1 }} />
        {Array.from({ length: rows }).map((_, idx) => (
          <Skeleton key={idx} variant="rounded" height={height} sx={{ borderRadius: 1, opacity: 1 - idx * 0.12 }} />
        ))}
      </Stack>
    </Box>
  );
}

TableSkeleton.propTypes = {
  columns: PropTypes.number,
  rows: PropTypes.number,
  height: PropTypes.number,
  sx: PropTypes.object,
};
