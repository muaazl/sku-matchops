import React from 'react';
import PropTypes from 'prop-types';
import { Box, Divider, Typography } from '@mui/material';
import { fmtTime, formatGk } from '../../utils/formatters';
import { buildSkuAuditMetadataRows } from '../../utils/skuExportUtils';
import RulesAppliedList from '../common/RulesAppliedList';
import { SideDrawer } from '../common/SideDrawer';

function DetailPairsList({ pairs }) {
  return pairs.map(([k, v]) => (
    <Box key={k} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.7 }}>
      <Typography variant="body2" color="text.secondary">
        {k}
      </Typography>
      <Typography variant="body2" sx={{ fontWeight: 500, textAlign: 'right', maxWidth: '70%' }}>
        {v}
      </Typography>
    </Box>
  ));
}

export function SkuDetailDrawer({ selectedSku, selectedJob, onClose }) {
  const isClassifier = selectedJob?.task?.toLowerCase() === 'classifier';
  const isMatcher = selectedJob?.task?.toLowerCase() === 'matcher';

  const summaryPairs = selectedSku
    ? [
        ['Domain', selectedSku.domain],
        ['Generic Keywords', formatGk(selectedSku.generic_keywords || selectedSku.gk_json)],
        ['Basic Type', selectedSku.basic_type || selectedSku.bt],
        ...(selectedSku.domain === 'food'
          ? [['Region', selectedSku.region]]
          : [['Categories', selectedSku.categories || selectedSku.region]]),
        ['Processed at', fmtTime(selectedSku.created_at)],
      ].filter(([, v]) => v !== null && v !== undefined)
    : [];

  const inputPairs = selectedSku
    ? [
        ['Price', selectedSku.input_price != null ? Number(selectedSku.input_price).toFixed(2) : null],
        ['Description', selectedSku.input_description],
        ['Category', selectedSku.input_category],
      ].filter(([, v]) => v !== null && v !== undefined && v !== '')
    : [];

  const auditPairs = selectedSku ? buildSkuAuditMetadataRows(selectedSku, { isClassifier, isMatcher }) : [];

  return (
    <SideDrawer open={!!selectedSku} onClose={onClose} title={selectedSku?.sku_name || ''}>
      {selectedSku && (
        <Box>
          <Typography variant="caption" color="text.secondary">
            {selectedSku.id} · {selectedSku.batch_id}
          </Typography>
          <Divider sx={{ my: 2 }} />
          <DetailPairsList pairs={summaryPairs} />

          {inputPairs.length > 0 && (
            <>
              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Input Details
              </Typography>
              <DetailPairsList pairs={inputPairs} />
            </>
          )}

          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            Audit Details
          </Typography>
          <DetailPairsList pairs={auditPairs} />

          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            Rules applied
          </Typography>
          <RulesAppliedList rules={selectedSku.rules_applied_json} />
        </Box>
      )}
    </SideDrawer>
  );
}

SkuDetailDrawer.propTypes = {
  selectedSku: PropTypes.object,
  selectedJob: PropTypes.object,
  onClose: PropTypes.func.isRequired,
};
