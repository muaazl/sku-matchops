import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, Chip, Grid, Stack, Typography } from '@mui/material';
import { StatusChip } from '../common/StatusChip';

export function AuditSummaryCard({ auditTrace, sku, domain, auditTask, isClassifierAudit }) {
  return (
    <Card sx={{ borderLeft: '4px solid', borderColor: 'primary.main' }}>
      <CardContent sx={{ py: 2.5 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid size={{ xs: 12, md: 4 }}>
            <Typography variant="caption" color="text.secondary" fontWeight={600}>
              TARGET SKU NAME
            </Typography>
            <Typography variant="h6" fontWeight={700}>
              {auditTrace.input?.sku_name || sku}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Chip
                label={`Domain: ${(auditTrace.input?.domain || domain).toUpperCase()}`}
                size="small"
                color="primary"
              />
              <Chip label={`Task: ${auditTask}`} size="small" variant="outlined" />
            </Box>
          </Grid>

          {isClassifierAudit ? (
            <Grid size={{ xs: 12, md: 4 }}>
              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                PREDICTED BASIC TYPE &amp; REGION
              </Typography>
              <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body1" fontWeight={600}>
                    {auditTrace.final_output?.suggested_bt || '(None)'}
                  </Typography>
                  <StatusChip status={auditTrace.final_output?.bt_status || 'LOW'} />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  <strong>{auditTrace.stage4_third_tag_classification?.tag_label || 'Category'}:</strong>{' '}
                  {auditTrace.final_output?.suggested_region || '-'}
                </Typography>
              </Stack>
            </Grid>
          ) : (
            <Grid size={{ xs: 12, md: 4 }}>
              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                MATCHED CATALOG ITEM
              </Typography>
              <Typography variant="subtitle1" fontWeight={600}>
                {auditTrace.final_output?.matched_catalog_name || '(No Match)'}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
                <StatusChip status={auditTrace.final_output?.status || 'Low / Rejected'} />
                <Typography variant="caption" color="text.secondary">
                  Score: {auditTrace.final_output?.score != null ? auditTrace.final_output.score.toFixed(2) : '0.0'}
                </Typography>
              </Box>
            </Grid>
          )}

          <Grid size={{ xs: 12, md: 4 }}>
            <Typography variant="caption" color="text.secondary" fontWeight={600}>
              FINAL TAGS PRODUCED
            </Typography>
            <Stack spacing={0.5} sx={{ mt: 0.5 }}>
              <Typography variant="body2">
                <strong>BT:</strong> {auditTrace.final_output?.suggested_bt || '-'}
              </Typography>
              <Typography variant="body2">
                <strong>GK:</strong> {auditTrace.final_output?.suggested_gk || '-'}
              </Typography>
              <Typography variant="body2">
                <strong>Region/Cat:</strong> {auditTrace.final_output?.suggested_region || '-'}
              </Typography>
            </Stack>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

AuditSummaryCard.propTypes = {
  auditTrace: PropTypes.object.isRequired,
  sku: PropTypes.string,
  domain: PropTypes.string,
  auditTask: PropTypes.string.isRequired,
  isClassifierAudit: PropTypes.bool.isRequired,
};
