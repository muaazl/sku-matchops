import React from 'react';
import PropTypes from 'prop-types';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

export function CandidatePoolTable({ candidates, showRawScore = false }) {
  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 700 }}>#</TableCell>
            <TableCell sx={{ fontWeight: 700 }}>Catalog Name</TableCell>
            <TableCell sx={{ fontWeight: 700 }}>Brand</TableCell>
            <TableCell sx={{ fontWeight: 700 }}>BasicType</TableCell>
            {showRawScore && (
              <TableCell sx={{ fontWeight: 700 }} align="right">
                Raw CE Score
              </TableCell>
            )}
            <TableCell sx={{ fontWeight: 700 }} align="right">
              Final CE Score
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(candidates || []).map((c, idx) => (
            <TableRow key={idx}>
              <TableCell>{idx + 1}</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>{c.catalog_name}</TableCell>
              <TableCell>{c.brand || '-'}</TableCell>
              <TableCell>{c.basic_type || '-'}</TableCell>
              {showRawScore && <TableCell align="right">{c.raw_cross_score?.toFixed(4)}</TableCell>}
              <TableCell align="right" sx={{ fontWeight: 600 }}>
                {c.final_cross_score?.toFixed(4)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

CandidatePoolTable.propTypes = {
  candidates: PropTypes.array,
  showRawScore: PropTypes.bool,
};
