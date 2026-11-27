import React from 'react';
import PropTypes from 'prop-types';
import {
  Alert,
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  LinearProgress,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { useJobDetail } from '../../hooks/useJobDetail';
import { JOB_STAGE_LABELS, JOB_STAGES } from '../../utils/constants';
import { fmtEta, fmtTime } from '../../utils/formatters';
import { CardSkeleton } from '../common/CardSkeleton';
import { SectionCard } from '../common/SectionCard';
import { StatusChip } from '../common/StatusChip';

export default function JobDetailModal({ jobId, open, onClose }) {
  const { job, isLoading, isError, liveEta, liveDurationText } = useJobDetail({ jobId, open });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          border: (theme) => `1px solid ${theme.palette.divider}`,
        },
      }}
    >
      <DialogTitle sx={{ m: 0, p: 2.5, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontFamily: 'monospace' }}>
            Job {jobId}
          </Typography>
          {job?.status && <StatusChip status={job.status} />}
        </Box>
        <IconButton onClick={onClose} size="small" aria-label="Close job details">
          <Close sx={{ fontSize: 20 }} />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 3, backgroundColor: (theme) => theme.palette.background.default }}>
        {isLoading && (
          <Stack spacing={3}>
            <CardSkeleton height={200} />
            <CardSkeleton height={100} />
          </Stack>
        )}
        {isError && !job && (
          <Alert severity="error" sx={{ borderRadius: 2 }}>
            Job {jobId} not found.
          </Alert>
        )}

        {job && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <SectionCard title="Live progress" subheader="Execution stage & timeline">
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 2,
                  flexWrap: 'wrap',
                  gap: 1,
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Stage:{' '}
                  <Typography component="span" variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    {JOB_STAGE_LABELS[job.current_stage] || job.current_stage}
                  </Typography>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Progress:{' '}
                  <Typography component="span" variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    {Math.round(job.progress_pct || 0)}%
                  </Typography>{' '}
                  | ETA: {fmtEta(liveEta ?? job.eta_seconds)}
                </Typography>
              </Box>
              <Box sx={{ width: '100%', mb: 3 }}>
                <LinearProgress
                  variant={job.status === 'running' && (job.progress_pct || 0) === 0 ? 'indeterminate' : 'determinate'}
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
                <Alert severity="error" sx={{ mt: 3, borderRadius: 2 }}>
                  {job.error_message}
                </Alert>
              )}
            </SectionCard>

            <SectionCard title="Details" subheader="Timing and execution parameters">
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  flexWrap: 'wrap',
                  gap: 3,
                }}
              >
                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600, display: 'block' }}
                  >
                    Started
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                    {fmtTime(job.started_at)}
                  </Typography>
                </Box>
                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600, display: 'block' }}
                  >
                    Completed
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                    {fmtTime(job.completed_at)}
                  </Typography>
                </Box>
                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600, display: 'block' }}
                  >
                    Duration
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
                    {liveDurationText}
                  </Typography>
                </Box>
              </Box>
            </SectionCard>
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
