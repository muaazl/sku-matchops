import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { useChart } from '../../hooks/useChart';

export function VolumeTrendChart({ data }) {
  const theme = useTheme();
  const ref = useChart({
    type: 'bar',
    data: {
      labels: data.map((d) => d.label),
      datasets: [
        {
          type: 'bar',
          label: 'SKUs Processed',
          data: data.map((d) => d.skus),
          backgroundColor: theme.palette.primary.main + 'cc',
          borderRadius: 4,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Avg Confidence %',
          data: data.map((d) => d.avgConfidence),
          borderColor: theme.palette.success.main,
          backgroundColor: theme.palette.success.main,
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: theme.palette.text.secondary, boxWidth: 12, usePointStyle: true },
        },
        tooltip: { enabled: true },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: theme.palette.text.secondary },
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: { color: theme.palette.divider },
          ticks: { color: theme.palette.text.secondary },
          beginAtZero: true,
          title: { display: true, text: 'SKUs Count', color: theme.palette.text.secondary },
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: theme.palette.success.main, callback: (v) => `${v}%` },
          min: 0,
          max: 100,
          title: { display: true, text: 'Avg Confidence %', color: theme.palette.success.main },
        },
      },
    },
  });
  return <canvas ref={ref} />;
}

VolumeTrendChart.propTypes = {
  data: PropTypes.array.isRequired,
};
