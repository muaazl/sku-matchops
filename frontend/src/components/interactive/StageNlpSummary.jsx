import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, Grid, Paper, Typography } from '@mui/material';
import { extractAuditEntities } from '../../utils/auditUtils';

export function StageNlpSummary({ auditTrace }) {
  const entities = extractAuditEntities(auditTrace);
  const weight = auditTrace.stage1_nlp?.extracted_weights;

  return (
    <Box>
      <Typography variant="subtitle1" fontWeight={700} color="primary" sx={{ mb: 1.5 }}>
        Stage 1: NLP Normalization &amp; Entity Extraction
      </Typography>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, sm: 6 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Clean Input
            </Typography>
            <Typography variant="body1" fontWeight={600}>
              {auditTrace.stage1_nlp?.clean_input || '-'}
            </Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, sm: 6 }}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Weight Stripped Input
            </Typography>
            <Typography variant="body1" fontWeight={600}>
              {auditTrace.stage1_nlp?.weight_stripped_input || '-'}
            </Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Extracted Weight Value
            </Typography>
            <Typography variant="body1" fontWeight={600}>
              {weight?.value != null ? `${weight.value} ${weight.unit || ''}` : 'None'}
            </Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, sm: 8 }}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              NER Extracted Entities
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
              {entities.length > 0 ? (
                entities.map((item) => <Chip key={item.id} label={item.label} size="small" variant="outlined" />)
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No entity tokens detected
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

StageNlpSummary.propTypes = {
  auditTrace: PropTypes.object.isRequired,
};
