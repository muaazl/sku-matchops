import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
  Typography,
  Alert,
  Dialog,
  DialogContent,
  IconButton,
  DialogTitle,
} from '@mui/material';
import { X } from 'lucide-react';
import { JOB_STAGES, JOB_STAGE_LABELS } from '../constants';
import { fmtEta, fmtTime, fmtDuration } from '../utils';
import { useQuery } from '@tanstack/react-query';
import { getJob } from '../api';

export default function JobDetailModal({ jobId, open, onClose }) {
  const {
    data: job,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    enabled: !!jobId && open,
    refetchInterval: (query) => {
      const j = query.state.data;
      return j && (j.status === 'running' || j.status === 'queued') ? 2500 : false;
    },
  });

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Job Details: {jobId}
        <IconButton onClick={onClose} size="small" aria-label="Close job details">
          <X size={20} />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 3, backgroundColor: (theme) => theme.palette.background.default }}>
        {isLoading && <LinearProgress />}
        {isError && !job && <Alert severity="error">Job {jobId} not found.</Alert>}

        {job && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <Card>
              <CardHeader title="Live progress" titleTypographyProps={{ variant: 'subtitle1' }} />
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Stage:{' '}
                    <b style={{ color: 'inherit' }}>{JOB_STAGE_LABELS[job.current_stage] || job.current_stage}</b>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Progress: <b>{Math.round(job.progress_pct || 0)}%</b> | ETA: {fmtEta(job.eta_seconds)}
                  </Typography>
                </Box>
                <Box sx={{ width: '100%', mb: 3 }}>
                  <LinearProgress
                    variant={
                      job.status === 'running' && (job.progress_pct || 0) === 0 ? 'indeterminate' : 'determinate'
                    }
                    value={Math.min(100, Math.max(0, job.progress_pct || 0))}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                <Stepper
                  activeStep={JOB_STAGES.indexOf(job.current_stage)}
                  alternativeLabel
                  sx={{ flexWrap: 'wrap', rowGap: 2 }}
                >
                  {JOB_STAGES.map((s) => (
                    <Step key={s} completed={JOB_STAGES.indexOf(s) < JOB_STAGES.indexOf(job.current_stage)}>
                      <StepLabel>{JOB_STAGE_LABELS[s]}</StepLabel>
                    </Step>
                  ))}
                </Stepper>
                {job.error_message && (
                  <Alert severity="error" sx={{ mt: 3 }}>
                    {job.error_message}
                  </Alert>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader title="Details" titleTypographyProps={{ variant: 'subtitle1' }} />
              <CardContent sx={{ px: 3, pb: '16px !important' }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    flexWrap: 'wrap',
                    gap: 2,
                  }}
                >
                  <Box>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ textTransform: 'uppercase', letterSpacing: 0.4, display: 'block' }}
                    >
                      Started
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500 }}>
                      {fmtTime(job.started_at)}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ textTransform: 'uppercase', letterSpacing: 0.4, display: 'block' }}
                    >
                      Completed
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500 }}>
                      {fmtTime(job.completed_at)}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ textTransform: 'uppercase', letterSpacing: 0.4, display: 'block' }}
                    >
                      Duration
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500 }}>
                      {fmtDuration(job.duration_minutes)}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
}

JobDetailModal.propTypes = {
  jobId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
};
