import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { useChart } from '../../hooks/useChart';

export function ConfidenceDoughnutChart({ data }) {
  const theme = useTheme();
  const ref = useChart({
    type: 'doughnut',
    data: {
      labels: data.map((d) => d.tier),
      datasets: [
        {
          data: data.map((d) => d.count),
          backgroundColor: data.map(
            (d) =>
              d.color ||
              (d.tier.includes('High')
                ? theme.palette.success.main
                : d.tier.includes('Medium')
                ? theme.palette.warning.main
                : theme.palette.error.main)
          ),
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
            label: (ctx) => {
              const item = data[ctx.dataIndex];
              return ` ${item.tier}: ${item.count.toLocaleString()} SKUs (${item.pct}%)`;
            },
          },
        },
      },
      cutout: '70%',
    },
  });
  return <canvas ref={ref} />;
}

ConfidenceDoughnutChart.propTypes = {
  data: PropTypes.array.isRequired,
};
