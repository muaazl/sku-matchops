import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

function useChart(config) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return undefined;
    chartRef.current = new Chart(canvasRef.current, config);
    return () => {
      chartRef.current?.destroy();
    };
  }, [JSON.stringify(config.data), JSON.stringify(config.options), config.type]);

  return canvasRef;
}

const baseOptions = (theme) => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: theme.palette.text.secondary, boxWidth: 12, boxHeight: 12, usePointStyle: true },
    },
    tooltip: { enabled: true },
  },
  scales: {
    x: {
      grid: { color: theme.palette.divider, display: false },
      ticks: { color: theme.palette.text.secondary },
    },
    y: {
      grid: { color: theme.palette.divider },
      ticks: { color: theme.palette.text.secondary },
      beginAtZero: true,
    },
  },
});

export function JobsBarChart({ data }) {
  const theme = useTheme();
  const ref = useChart({
    type: 'bar',
    data: {
      labels: data.map((d) => d.day),
      datasets: [
        {
          label: 'Completed',
          data: data.map((d) => d.completed),
          backgroundColor: theme.palette.success.main,
          borderRadius: 4,
        },
        {
          label: 'Failed',
          data: data.map((d) => d.failed),
          backgroundColor: theme.palette.error.main,
          borderRadius: 4,
        },
      ],
    },
    options: {
      ...baseOptions(theme),
      scales: {
        ...baseOptions(theme).scales,
        x: { ...baseOptions(theme).scales.x, stacked: true },
        y: { ...baseOptions(theme).scales.y, stacked: true },
      },
    },
  });
  return <canvas ref={ref} />;
}

JobsBarChart.propTypes = { data: PropTypes.array.isRequired };

export function RequestVolumeChart({ data }) {
  const theme = useTheme();
  const ref = useChart({
    type: 'line',
    data: {
      labels: data.map((d) => d.hour),
      datasets: [
        {
          label: 'Requests / hr',
          data: data.map((d) => d.requests),
          borderColor: theme.palette.primary.main,
          backgroundColor: theme.palette.primary.main + '22',
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: baseOptions(theme),
  });
  return <canvas ref={ref} />;
}

RequestVolumeChart.propTypes = { data: PropTypes.array.isRequired };

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

ConfidenceDoughnutChart.propTypes = { data: PropTypes.array.isRequired };

export function MatchSourcePieChart({ data }) {
  const theme = useTheme();
  const palette = [
    theme.palette.primary.main,
    theme.palette.success.main,
    theme.palette.secondary.main,
    theme.palette.info.main,
    theme.palette.warning.main,
  ];

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

MatchSourcePieChart.propTypes = { data: PropTypes.array.isRequired };

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

VolumeTrendChart.propTypes = { data: PropTypes.array.isRequired };
