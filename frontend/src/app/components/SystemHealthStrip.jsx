import React from 'react';
import PropTypes from 'prop-types';
import { Box, Tooltip, Typography } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { getSystemStatus } from '../api';

function Counter({ label, value, hint, tone }) {
  return (
    <Tooltip title={hint || ''}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.75, whiteSpace: 'nowrap' }}>
        <Typography
          component="span"
          sx={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}
          color="text.secondary"
        >
          {label}
        </Typography>
        <span
          className="portal-mono"
          style={{ fontWeight: 600, color: tone === 'bad' ? 'var(--portal-bad-text)' : undefined }}
        >
          {value}
        </span>
      </Box>
    </Tooltip>
  );
}

// Live operational counters (GET /system/status), scoped to the active tenant.
export default function SystemHealthStrip() {
  const { data, isError } = useQuery({
    queryKey: ['system-status'],
    queryFn: getSystemStatus,
    refetchInterval: 15000,
  });

  const depth = data?.queue_depth;
  const queueHint = !data?.queue_online
    ? 'Redis queue offline: jobs fall back to direct engine dispatch'
    : depth
    ? `${depth.inflight} in flight, ${depth.delayed} waiting to retry (all tenants)`
    : 'Queue depth spans all tenants: platform admins only';
  const fmt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString());

  return (
    <Box
      role="status"
      aria-label="System health"
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: { xs: 2, md: 3.5 },
        px: { xs: 2, md: 4 },
        py: 1,
        borderBottom: '1px solid var(--portal-border)',
        backgroundColor: 'background.paper',
      }}
    >
      {isError ? (
        <Typography variant="caption" color="error">
          System status unavailable
        </Typography>
      ) : (
        <>
          <Counter label="Tenant" value={data?.tenant_id ?? '—'} />
          <Counter label="Processed SKUs" value={fmt(data?.processed_skus)} />
          <Counter
            label="Realtime queue"
            value={data?.queue_online ? fmt(depth?.realtime) : 'offline'}
            hint={queueHint}
          />
          <Counter label="Bulk queue" value={data?.queue_online ? fmt(depth?.bulk) : 'offline'} hint={queueHint} />
          <Counter
            label="DLQ"
            value={fmt(data?.dlq_size)}
            tone={data?.dlq_size ? 'bad' : undefined}
            hint="Dead-lettered jobs for this tenant"
          />
          <Counter
            label="Rule version"
            value={data?.rule_version ?? '—'}
            hint="Fingerprint of this tenant's active rules; stored with every job's results"
          />
        </>
      )}
    </Box>
  );
}

Counter.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  hint: PropTypes.string,
  tone: PropTypes.oneOf(['bad']),
};
