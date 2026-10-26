import React from 'react';
import { render, screen } from '@testing-library/react';
import { ConfidencePill, HttpStatusChip, StatusChip, statusTone } from './ui';

describe('statusTone', () => {
  it.each([
    ['completed', 'ok'],
    ['High Confidence', 'ok'],
    ['AUTO', 'ok'],
    ['active', 'ok'],
    ['REVIEW', 'warn'],
    ['Medium Confidence', 'warn'],
    ['pending', 'warn'],
    ['failed', 'bad'],
    ['Low / Rejected', 'bad'], // worst tone wins over any other keyword
    ['dlq', 'bad'],
    ['queued', 'neutral'],
    ['running', 'neutral'],
    ['inactive', 'neutral'],
  ])('%s -> %s', (status, tone) => {
    expect(statusTone(status)).toBe(tone);
  });
});

describe('pills', () => {
  it('renders the semantic pill class', () => {
    render(<StatusChip status="failed" />);
    expect(screen.getByText('failed').className).toContain('portal-pill--bad');
  });

  it.each([
    [0.92, 'High', 'ok', '92%'],
    [85, 'High', 'ok', '85%'], // 0-100 scale is normalised
    [0.6, 'Review', 'warn', '60%'],
    [0.59, 'Low', 'bad', '59%'],
  ])('confidence %s -> %s', (value, label, tone, pct) => {
    const { container } = render(<ConfidencePill value={value} />);
    expect(container.firstChild.className).toContain(`portal-pill--${tone}`);
    expect(container.textContent).toBe(`${label}${pct}`);
  });

  it('handles a missing confidence', () => {
    render(<ConfidencePill value={null} />);
    expect(screen.getByText('n/a').className).toContain('portal-pill--neutral');
  });

  it('colors HTTP codes by class', () => {
    render(<HttpStatusChip code={503} />);
    expect(screen.getByText('503').className).toContain('portal-pill--bad');
  });
});
