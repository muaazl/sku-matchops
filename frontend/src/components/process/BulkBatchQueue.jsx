import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Card, CardContent, Chip, CircularProgress, Stack, TextField, Typography } from '@mui/material';
import { PageContainer } from '../common/PageContainer';
import { PageHeader } from '../common/PageHeader';

export function BulkBatchQueue({ pendingJobs, onUpdateJob, onStartJob, onRemoveJob, onStartAll, onDone }) {
  const anyPending = pendingJobs.some((j) => j.status === 'pending' || j.status === 'error');

  return (
    <PageContainer>
      <PageHeader
        title="Bulk SKU Jobs"
        subtitle="Review each file/sheet below, then start them individually or all at once."
      />

      <Stack spacing={2} sx={{ mt: 3 }}>
        {pendingJobs.map((job) => (
          <Card key={job.id} variant="outlined">
            <CardContent>
              <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
                <Box sx={{ flexGrow: 1, minWidth: 200 }}>
                  <TextField
                    value={job.name}
                    onChange={(e) => onUpdateJob(job.id, { name: e.target.value })}
                    label="Job name"
                    size="small"
                    fullWidth
                    disabled={job.status === 'starting' || job.status === 'started'}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                    {job.sourceLabel} · {job.rows.length} SKU{job.rows.length === 1 ? '' : 's'}
                  </Typography>
                </Box>

                {job.status === 'started' ? (
                  <Chip label={`Queued — Job ${job.jobId}`} color="success" size="small" />
                ) : job.status === 'error' ? (
                  <Chip label={job.error || 'Failed'} color="error" size="small" sx={{ maxWidth: 320 }} />
                ) : job.status === 'starting' ? (
                  <Chip icon={<CircularProgress size={12} color="inherit" />} label="Starting…" size="small" />
                ) : job.rows.length === 0 ? (
                  <Chip label="No rows" color="warning" size="small" />
                ) : null}

                <Stack direction="row" spacing={1}>
                  <Button
                    size="small"
                    variant="contained"
                    disabled={job.status === 'starting' || job.status === 'started' || job.rows.length === 0}
                    onClick={() => onStartJob(job)}
                  >
                    {job.status === 'error' ? 'Retry' : 'Start'}
                  </Button>
                  <Button
                    size="small"
                    color="inherit"
                    variant="outlined"
                    disabled={job.status === 'starting'}
                    onClick={() => onRemoveJob(job.id)}
                  >
                    {job.status === 'started' ? 'Dismiss' : 'Remove'}
                  </Button>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        ))}

        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1.5, pt: 1 }}>
          <Button color="inherit" onClick={onDone}>
            Done
          </Button>
          <Button variant="contained" disabled={!anyPending} onClick={onStartAll}>
            Start All
          </Button>
        </Box>
      </Stack>
    </PageContainer>
  );
}

BulkBatchQueue.propTypes = {
  pendingJobs: PropTypes.array.isRequired,
  onUpdateJob: PropTypes.func.isRequired,
  onStartJob: PropTypes.func.isRequired,
  onRemoveJob: PropTypes.func.isRequired,
  onStartAll: PropTypes.func.isRequired,
  onDone: PropTypes.func.isRequired,
};
