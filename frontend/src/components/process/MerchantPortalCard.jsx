import React from 'react';
import PropTypes from 'prop-types';
import { Button, CircularProgress, Stack, TextField } from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import { DomainSelector, TaskSelector } from '../common/DomainTaskSelectors';
import { SectionCard } from '../common/SectionCard';

export function MerchantPortalCard({
  merchantId,
  onMerchantIdChange,
  domain,
  onDomainChange,
  task,
  onTaskChange,
  fetchingPortal,
  onFetch,
}) {
  const canSubmit = merchantId.trim() !== '';

  return (
    <SectionCard title="Portal Details" subheader="Fetch live merchant catalog via portal integration">
      <Stack spacing={2.5}>
        <TextField
          label="Merchant ID"
          value={merchantId}
          onChange={(e) => onMerchantIdChange(e.target.value)}
          placeholder="e.g. 8842"
          fullWidth
        />

        <DomainSelector value={domain} onChange={onDomainChange} ariaLabel="Select domain" />
        <TaskSelector value={task} onChange={onTaskChange} ariaLabel="Select matching task" />

        <Button
          variant="contained"
          startIcon={
            fetchingPortal ? <CircularProgress size={16} color="inherit" /> : <PlayArrow sx={{ fontSize: 16 }} />
          }
          onClick={onFetch}
          disabled={!canSubmit || fetchingPortal}
          fullWidth
          size="large"
          sx={{ mt: 1, height: 42, fontWeight: 600 }}
        >
          {fetchingPortal ? 'Fetching...' : 'Fetch & Process SKUs'}
        </Button>
      </Stack>
    </SectionCard>
  );
}

MerchantPortalCard.propTypes = {
  merchantId: PropTypes.string.isRequired,
  onMerchantIdChange: PropTypes.func.isRequired,
  domain: PropTypes.string.isRequired,
  onDomainChange: PropTypes.func.isRequired,
  task: PropTypes.string.isRequired,
  onTaskChange: PropTypes.func.isRequired,
  fetchingPortal: PropTypes.bool.isRequired,
  onFetch: PropTypes.func.isRequired,
};
