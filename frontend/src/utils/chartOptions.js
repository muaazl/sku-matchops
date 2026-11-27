export function createBaseChartOptions(theme) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: theme.palette.text.secondary,
          boxWidth: 12,
          boxHeight: 12,
          usePointStyle: true,
        },
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
  };
}

export function getChartSeriesPalette(theme) {
  return [
    theme.palette.primary.main,
    theme.palette.success.main,
    theme.palette.secondary.main,
    theme.palette.info.main,
    theme.palette.warning.main,
    theme.palette.error.main,
  ];
}
