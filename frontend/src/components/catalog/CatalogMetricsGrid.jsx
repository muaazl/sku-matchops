import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, Typography } from '@mui/material';
import { getCatalogMetricCards } from '../../utils/catalogUtils';

export function CatalogMetricsGrid({ domain, activeDataset, summaryStats, onSelectDataset }) {
  const cards = getCatalogMetricCards(domain);

  return (
    <Box
      sx={{
        display: 'flex',
        width: '100%',
        gap: 2,
        mb: 4,
        flexWrap: 'wrap',
        '& > *': {
          flex: '1 1 0px',
          minWidth: { xs: 'calc(50% - 16px)', sm: 'calc(33.33% - 16px)', md: '0px' },
        },
      }}
    >
      {cards.map((card) => {
        const isSelected = activeDataset === card.id;
        const statValue = summaryStats?.[card.statKey];
        return (
          <Card
            key={card.id}
            sx={{
              cursor: 'pointer',
              border: isSelected ? '2px solid' : '1px solid',
              borderColor: isSelected ? 'primary.main' : 'divider',
              bgcolor: isSelected ? 'action.selected' : 'background.paper',
              '&:hover': { borderColor: 'primary.main' },
            }}
            onClick={() => onSelectDataset(card.id)}
          >
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography color="text.secondary" variant="caption" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
                {card.label}
              </Typography>
              <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>
                {statValue != null ? statValue.toLocaleString() : '---'}
              </Typography>
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
}

CatalogMetricsGrid.propTypes = {
  domain: PropTypes.string.isRequired,
  activeDataset: PropTypes.string.isRequired,
  summaryStats: PropTypes.object,
  onSelectDataset: PropTypes.func.isRequired,
};
