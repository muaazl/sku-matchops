import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, Divider, Typography } from '@mui/material';
import { SideDrawer } from '../common/SideDrawer';

export function PointPayloadDrawer({ payloadPoint, onClose }) {
  const scoreDisplay = payloadPoint?.score != null ? Number(payloadPoint.score).toFixed(6) : 'N/A';

  return (
    <SideDrawer open={!!payloadPoint} onClose={onClose} title="Point Details" width={500}>
      {payloadPoint && (
        <Box>
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Point ID
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {payloadPoint.id}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Chip size="small" label={`Score: ${scoreDisplay}`} color="primary" />
            </Box>
          </Box>

          <Divider sx={{ mb: 2 }} />

          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
            Payload
          </Typography>
          <Box
            sx={{ flexGrow: 1, overflow: 'auto', bgcolor: 'grey.900', color: 'common.white', p: 2, borderRadius: 1 }}
          >
            <pre
              style={{
                margin: 0,
                fontFamily: 'monospace',
                fontSize: '0.825rem',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {JSON.stringify(payloadPoint.payload, null, 2)}
            </pre>
          </Box>
        </Box>
      )}
    </SideDrawer>
  );
}

PointPayloadDrawer.propTypes = {
  payloadPoint: PropTypes.object,
  onClose: PropTypes.func.isRequired,
};
