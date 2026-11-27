import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, Typography } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { CONDITION_LABELS } from '../../utils/constants';

export function RuleSummaryPreview({ conditionsCount, conditionGroups }) {
  return (
    <Box
      sx={{
        mb: 2,
        p: 1.25,
        borderRadius: 1.5,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: (theme) => alpha(theme.palette.primary.main, 0.04),
      }}
    >
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: 'block', fontWeight: 600, mb: 0.5, textTransform: 'uppercase', letterSpacing: 0.5 }}
      >
        Rule summary
      </Typography>
      {conditionsCount === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No conditions — matches every SKU in this domain.
        </Typography>
      ) : (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 0.6, lineHeight: 1.8 }}>
          <Chip label="IF" size="small" sx={{ height: 20, fontSize: '0.68rem', fontWeight: 700, borderRadius: 1 }} />
          {conditionGroups.map((group, gIdx) => (
            <React.Fragment key={`preview-grp-${gIdx}`}>
              {gIdx > 0 && (
                <Chip
                  label="OR"
                  size="small"
                  color="warning"
                  sx={{ height: 20, fontSize: '0.68rem', fontWeight: 700, borderRadius: 1 }}
                />
              )}
              {conditionGroups.length > 1 && group.length > 1 && (
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
                  (
                </Typography>
              )}
              {group.map(({ cond, index }, itemIdx) => (
                <React.Fragment key={`preview-cond-${index}`}>
                  {itemIdx > 0 && (
                    <Chip
                      label="AND"
                      size="small"
                      color="primary"
                      variant="outlined"
                      sx={{ height: 20, fontSize: '0.65rem', fontWeight: 700, borderRadius: 1 }}
                    />
                  )}
                  <Typography component="span" variant="body2" sx={{ fontSize: '0.8rem' }}>
                    {!!cond.negate && (
                      <Box component="span" sx={{ color: 'error.main', fontWeight: 700, mr: 0.5 }}>
                        NOT
                      </Box>
                    )}
                    <Box component="span" sx={{ fontWeight: 500 }}>
                      {CONDITION_LABELS[cond.condition_type] || cond.condition_type}
                    </Box>{' '}
                    <Box
                      component="span"
                      sx={{
                        fontWeight: 600,
                        color: cond.value ? 'text.primary' : 'text.disabled',
                        fontStyle: cond.value ? 'normal' : 'italic',
                      }}
                    >
                      {cond.value ? `"${cond.value}"` : '(empty)'}
                    </Box>
                  </Typography>
                </React.Fragment>
              ))}
              {conditionGroups.length > 1 && group.length > 1 && (
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
                  )
                </Typography>
              )}
            </React.Fragment>
          ))}
        </Box>
      )}
    </Box>
  );
}

RuleSummaryPreview.propTypes = {
  conditionsCount: PropTypes.number.isRequired,
  conditionGroups: PropTypes.array.isRequired,
};
