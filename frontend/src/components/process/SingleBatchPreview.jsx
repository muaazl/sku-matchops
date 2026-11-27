import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Grid,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import { PageContainer } from '../common/PageContainer';
import { PageHeader } from '../common/PageHeader';

export function SingleBatchPreview({ previewData, jobName, onJobNameChange, isCreatingBatch, onCancel, onStart }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    setPage(0);
  }, [previewData]);

  const paginatedRows = previewData.rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <PageContainer>
      <PageHeader
        title="Confirm SKU Job"
        subtitle="Review and customize parsed SKU details before starting the batch processing job."
      />

      <Box sx={{ mt: 3 }}>
        <Card>
          <CardHeader
            title="Job Settings & Data Confirmation"
            titleTypographyProps={{ variant: 'h6', sx: { fontWeight: 600 } }}
            subheader={`Total SKU Items: ${previewData.rows.length}`}
          />
          <CardContent>
            <Stack spacing={3}>
              <Grid container spacing={2} alignItems="center">
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    label="Job / Sheet Name"
                    value={jobName}
                    onChange={(e) => onJobNameChange(e.target.value)}
                    placeholder="e.g. mrc_8842"
                    fullWidth
                    required
                    error={!jobName.trim()}
                    helperText={!jobName.trim() ? 'Job name is required' : ''}
                  />
                </Grid>
                <Grid size={{ xs: 12, md: 6 }}>
                  <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                    <Chip
                      label={`Source: ${previewData.source}`}
                      color="info"
                      variant="outlined"
                      sx={{ textTransform: 'capitalize' }}
                    />
                    <Chip
                      label={`Domain: ${previewData.domain}`}
                      color="primary"
                      variant="outlined"
                      sx={{ textTransform: 'capitalize' }}
                    />
                    <Chip
                      label={`Task: ${previewData.task}`}
                      color="secondary"
                      variant="outlined"
                      sx={{ textTransform: 'capitalize' }}
                    />
                  </Stack>
                </Grid>
              </Grid>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                Parsed SKUs Preview
              </Typography>

              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 440 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>SKU Name</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Category</TableCell>
                      <TableCell sx={{ fontWeight: 600 }} align="right">
                        Price
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {paginatedRows.map((row, idx) => (
                      <TableRow key={idx} hover>
                        <TableCell sx={{ fontWeight: 500 }}>{row.name || '—'}</TableCell>
                        <TableCell>{row.category || '—'}</TableCell>
                        <TableCell align="right">
                          {row.price !== undefined ? Number.parseFloat(row.price).toFixed(2) : '0.00'}
                        </TableCell>
                        <TableCell
                          sx={{
                            maxWidth: 250,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {row.description || '—'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <TablePagination
                rowsPerPageOptions={[5, 10, 25, 50]}
                component="div"
                count={previewData.rows.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={(e, newPage) => setPage(newPage)}
                onRowsPerPageChange={(e) => {
                  setRowsPerPage(Number.parseInt(e.target.value, 10));
                  setPage(0);
                }}
              />

              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: 2,
                  pt: 2,
                  borderTop: 1,
                  borderColor: 'divider',
                }}
              >
                <Button variant="outlined" color="inherit" onClick={onCancel} disabled={isCreatingBatch}>
                  Cancel
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={
                    isCreatingBatch ? (
                      <CircularProgress size={16} color="inherit" />
                    ) : (
                      <PlayArrow sx={{ fontSize: 16 }} />
                    )
                  }
                  onClick={onStart}
                  disabled={!jobName.trim() || isCreatingBatch}
                  sx={{ px: 4, fontWeight: 600 }}
                >
                  {isCreatingBatch ? 'Starting...' : 'Start Processing'}
                </Button>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      </Box>
    </PageContainer>
  );
}

SingleBatchPreview.propTypes = {
  previewData: PropTypes.object.isRequired,
  jobName: PropTypes.string.isRequired,
  onJobNameChange: PropTypes.func.isRequired,
  isCreatingBatch: PropTypes.bool.isRequired,
  onCancel: PropTypes.func.isRequired,
  onStart: PropTypes.func.isRequired,
};
