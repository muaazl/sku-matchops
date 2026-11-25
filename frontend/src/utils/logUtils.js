export function parseLogLines(logsText, filterQuery = '') {
  if (!logsText) return [];

  const rawLines = logsText.split('\n');
  const needle = filterQuery ? filterQuery.toLowerCase() : '';
  const filtered = needle ? rawLines.filter((line) => line.toLowerCase().includes(needle)) : rawLines;

  return filtered.map((line, idx) => {
    let color = '#d4d4d4';
    let fontWeight = 'normal';

    if (line.includes(' - ERROR - ') || line.includes(' - CRITICAL - ')) {
      color = '#f87171';
      fontWeight = 'bold';
    } else if (line.includes(' - WARNING - ')) {
      color = '#fbbf24';
      fontWeight = 'bold';
    } else if (line.includes(' - INFO - ')) {
      color = '#34d399';
    } else if (line.includes(' - DEBUG - ')) {
      color = '#9ca3af';
    }

    return { text: line, color, fontWeight, id: idx };
  });
}

export function countNonEmptyLogLines(logsText) {
  if (!logsText) return 0;
  return logsText.split('\n').filter(Boolean).length;
}
