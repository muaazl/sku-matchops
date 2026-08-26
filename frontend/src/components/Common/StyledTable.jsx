import { alpha, styled } from '@mui/material/styles';
import { TableBody, TableCell, TableContainer, TableHead, TableRow, Skeleton } from '@mui/material';
import { Link } from 'react-router-dom';
import React from 'react';
import PropTypes from 'prop-types';

export const StyledTableContainer = styled(TableContainer)(({ theme }) => ({
  borderRadius: '0.5rem',
  border: `1px solid ${alpha(theme.palette.divider, 0.12)}`,
}));

export const StyledTableHead = styled(TableHead)(({ theme }) => ({
  backgroundColor: theme.palette.background.default,
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.12)}`,
}));

export const StyledHeaderCell = styled(TableCell)(({ theme }) => ({
  padding: '1rem',
  fontSize: '0.875rem',
  fontWeight: 600,
  lineHeight: 1.5,
  color: theme.palette.text.secondary,
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  whiteSpace: 'nowrap',
}));

export const StyledTableBody = styled(TableBody)(({ theme }) => ({
  backgroundColor: theme.palette.background.paperElevation1,
}));

export const StyledTableRow = styled(TableRow)(({ theme }) => ({
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.12)}`,
  '&:last-child': {
    borderBottom: 'none',
  },
  '& .MuiTableCell-root': {
    padding: '1rem',
    borderBottom: 'none',
    fontSize: '1rem',
    fontWeight: 400,
    lineHeight: 1.5,
    color: theme.palette.text.primary,
  },
}));

export const StyledLink = styled(Link)({
  textDecoration: 'none',
  color: 'inherit',
  '&:hover': {
    textDecoration: 'underline',
    textDecorationThickness: '1px',
    textUnderlineOffset: '2px',
  },
});

export const TableSkeleton = ({ columns, rows = 5 }) => {
  return (
    <>
      {[...Array(rows)].map((_, rowIndex) => (
        <StyledTableRow key={rowIndex}>
          {[...Array(columns)].map((_, colIndex) => (
            <TableCell key={colIndex}>
              <Skeleton variant="text" width={colIndex === 0 ? '40%' : '80%'} height={24} />
            </TableCell>
          ))}
        </StyledTableRow>
      ))}
    </>
  );
};

TableSkeleton.propTypes = {
  columns: PropTypes.number.isRequired,
  rows: PropTypes.number,
};
