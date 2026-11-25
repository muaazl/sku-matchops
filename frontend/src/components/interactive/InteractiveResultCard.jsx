import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Card, CardContent, CardHeader, CircularProgress, Divider, Typography } from '@mui/material';
import { Memory } from '@mui/icons-material';
import { ConfidenceBar } from '../common/ConfidenceBar';
import RulesAppliedList from '../common/RulesAppliedList';

export function InteractiveResultCard({ item, resolvedDomain, auditLoading, onAudit }) {
  return (
    <Card>
      <CardHeader
        title="Result"
        titleTypographyProps={{ variant: 'subtitle1', sx: { fontWeight: 600 } }}
        action={
          <Button
            variant="contained"
            color="secondary"
            size="small"
            startIcon={auditLoading ? <CircularProgress size={14} color="inherit" /> : <Memory sx={{ fontSize: 14 }} />}
            onClick={onAudit}
            disabled={auditLoading}
            sx={{ fontWeight: 600, textTransform: 'none', borderRadius: 2 }}
          >
            {auditLoading ? 'Auditing...' : 'Audit Result'}
          </Button>
        }
      />
      <CardContent>
        {item.matched_catalog_name && (
          <>
            <Typography variant="caption" color="text.secondary">
              Matched catalog item
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 500, mb: 1 }}>
              {item.matched_catalog_name}
            </Typography>
            {item.score != null && <ConfidenceBar label="Match score" value={item.score} status={item.status} />}
            {item.logic_notes && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {item.logic_notes}
              </Typography>
            )}
            <Divider sx={{ my: 2 }} />
          </>
        )}
        {item.suggested_bt && (
          <ConfidenceBar
            label={`Basic Type · ${item.suggested_bt}`}
            value={item.bt_confidence}
            status={item.bt_status}
            source={item.bt_source}
          />
        )}
        {item.suggested_gk && (
          <ConfidenceBar
            label={`Generic Keywords · ${item.suggested_gk}`}
            value={item.gk_confidence}
            status={item.gk_status}
          />
        )}
        {item.suggested_region && (
          <ConfidenceBar
            label={`${resolvedDomain === 'food' ? 'Region' : 'Categories'} · ${item.suggested_region}`}
            value={item.region_confidence}
            status={item.region_status}
            source={item.region_source}
          />
        )}
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Rules applied
          </Typography>
          <RulesAppliedList rules={item.rules_applied} />
        </Box>
      </CardContent>
    </Card>
  );
}

InteractiveResultCard.propTypes = {
  item: PropTypes.object.isRequired,
  resolvedDomain: PropTypes.string.isRequired,
  auditLoading: PropTypes.bool.isRequired,
  onAudit: PropTypes.func.isRequired,
};
