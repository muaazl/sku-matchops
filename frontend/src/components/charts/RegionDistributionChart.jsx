import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { useChart } from '../../hooks/useChart';
import { getChartSeriesPalette } from '../../utils/chartOptions';

export function RegionDistributionChart({ data }) {
  const theme = useTheme();
  const palette = getChartSeriesPalette(theme);

  const ref = useChart({
    type: 'doughnut',
    data: {
      labels: data.map((d) => d.tag),
      datasets: [
        {
          data: data.map((d) => d.count),
          backgroundColor: data.map((_, i) => palette[i % palette.length]),
          borderWidth: 2,
          borderColor: theme.palette.background.paper,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: theme.palette.text.secondary, boxWidth: 10, usePointStyle: true, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} SKUs`,
          },
        },
      },
      cutout: '65%',
    },
  });
  return <canvas ref={ref} />;
}

RegionDistributionChart.propTypes = {
  data: PropTypes.array.isRequired,
};
