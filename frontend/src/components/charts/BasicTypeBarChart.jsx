import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { useChart } from '../../hooks/useChart';

export function BasicTypeBarChart({ data }) {
  const theme = useTheme();
  const ref = useChart({
    type: 'bar',
    data: {
      labels: data.map((d) => d.basicType),
      datasets: [
        {
          label: 'SKUs',
          data: data.map((d) => d.count),
          backgroundColor: theme.palette.info.main,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.raw.toLocaleString()} SKUs`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: theme.palette.divider },
          ticks: { color: theme.palette.text.secondary },
          beginAtZero: true,
        },
        y: {
          grid: { display: false },
          ticks: { color: theme.palette.text.secondary },
        },
      },
    },
  });
  return <canvas ref={ref} />;
}

BasicTypeBarChart.propTypes = {
  data: PropTypes.array.isRequired,
};
