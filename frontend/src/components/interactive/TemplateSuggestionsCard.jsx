import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, CardHeader, Chip, CircularProgress, Stack, Tooltip, Typography } from '@mui/material';
import { ArrowForward, AutoAwesome } from '@mui/icons-material';

export function TemplateSuggestionsCard({ suggestions, loading }) {
  if (loading) {
    return (
      <Card>
        <CardContent sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
          <CircularProgress size={20} />
          <Typography variant="body2" color="text.secondary">
            Searching for matching template SKU in catalog...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!suggestions || !suggestions.matched) {
    return null;
  }

  return (
    <Card sx={{ borderLeft: '4px solid', borderColor: 'info.main' }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoAwesome sx={{ fontSize: 18, color: '#0288d1' }} />
            <Typography variant="subtitle1" fontWeight={600}>
              Template-Based Suggestions
            </Typography>
          </Box>
        }
      />
      <CardContent sx={{ pt: 0 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Found a similar base item in the catalog. Swapped the flavor/brand to generate matched tag suggestions.
        </Typography>

        <Box sx={{ p: 1.5, bgcolor: 'action.hover', borderRadius: 1, mb: 2 }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Base Catalog Item Match
          </Typography>
          <Typography variant="body2" fontWeight={500}>
            {suggestions.base_sku}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8, mt: 0.5 }}>
            <Typography
              variant="caption"
              sx={{ px: 0.8, py: 0.2, bgcolor: 'action.selected', borderRadius: 0.5, fontWeight: 500 }}
            >
              {suggestions.base_entity}
            </Typography>
            <ArrowForward sx={{ fontSize: 12, color: 'text.secondary' }} />
            <Typography
              variant="caption"
              sx={{
                px: 0.8,
                py: 0.2,
                bgcolor: 'info.light',
                color: 'info.contrastText',
                borderRadius: 0.5,
                fontWeight: 500,
              }}
            >
              {suggestions.new_entity}
            </Typography>
          </Box>
        </Box>

        <Stack spacing={1.5}>
          {suggestions.suggested_bt && (
            <Box>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                Suggested Basic Type
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" fontWeight={500}>
                  {suggestions.suggested_bt}
                </Typography>
                {suggestions.bt_info && (
                  <Chip
                    label={
                      suggestions.bt_info.status === 'exact_dictionary_match'
                        ? 'Exact Dictionary'
                        : suggestions.bt_info.status === 'fuzzy_snapped'
                        ? `Fuzzy Snapped (from "${suggestions.bt_info.original}")`
                        : 'New / Unregistered'
                    }
                    size="small"
                    color={
                      suggestions.bt_info.status === 'exact_dictionary_match'
                        ? 'success'
                        : suggestions.bt_info.status === 'fuzzy_snapped'
                        ? 'info'
                        : 'warning'
                    }
                    variant="outlined"
                    sx={{ height: 20, fontSize: '0.65rem' }}
                  />
                )}
              </Box>
            </Box>
          )}

          {suggestions.gk_info && suggestions.gk_info.length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                Suggested Generic Keywords
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {suggestions.gk_info.map((info, idx) => (
                  <Tooltip
                    key={idx}
                    title={
                      info.status === 'exact_dictionary_match'
                        ? 'Exact match in dictionary'
                        : info.status === 'fuzzy_snapped'
                        ? `Fuzzy snapped to dictionary from: ${info.original} (${Math.round(
                            info.snap_score
                          )}% similarity)`
                        : `New unregistered tag: ${info.original} (capitalized according to casing exceptions)`
                    }
                  >
                    <Chip
                      label={info.suggested}
                      size="small"
                      color={
                        info.status === 'exact_dictionary_match'
                          ? 'success'
                          : info.status === 'fuzzy_snapped'
                          ? 'info'
                          : 'default'
                      }
                      variant="outlined"
                      sx={{
                        fontSize: '0.75rem',
                        borderStyle: info.status === 'new_unregistered' ? 'dashed' : 'solid',
                        borderColor: info.status === 'new_unregistered' ? 'warning.main' : undefined,
                      }}
                    />
                  </Tooltip>
                ))}
              </Box>
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

TemplateSuggestionsCard.propTypes = {
  suggestions: PropTypes.object,
  loading: PropTypes.bool.isRequired,
};
