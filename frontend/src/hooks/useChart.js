import { useEffect, useRef } from 'react';
import {
  ArcElement,
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  DoughnutController,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
} from 'chart.js';

Chart.register(
  ArcElement,
  BarController,
  BarElement,
  CategoryScale,
  DoughnutController,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip
);

export function useChart(config) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const configKey = JSON.stringify({
    type: config.type,
    data: config.data,
    options: config.options,
  });

  useEffect(() => {
    if (!canvasRef.current) return undefined;
    if (chartRef.current && chartRef.current.config?.type === config.type) {
      chartRef.current.data = config.data;
      chartRef.current.options = config.options;
      chartRef.current.update('none');
      return undefined;
    }
    chartRef.current?.destroy();
    chartRef.current = new Chart(canvasRef.current, config);
    return undefined;
  }, [configKey, config.type]);

  useEffect(
    () => () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    },
    []
  );

  return canvasRef;
}
