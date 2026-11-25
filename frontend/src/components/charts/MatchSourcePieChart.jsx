import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { useChart } from '../../hooks/useChart';
import { getChartSeriesPalette } from '../../utils/chartOptions';

export function MatchSourcePieChart({ data }) {
  const theme = useTheme();
  const palette = getChartSeriesPalette(theme).slice(0, 5);

  const ref = useChart({
    type: 'doughnut',
    data: {
      labels: data.map((d) => d.source),
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

MatchSourcePieChart.propTypes = {
  data: PropTypes.array.isRequired,
};
