import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import { vi } from 'vitest';
import * as api from '../api';
import { useAuth } from '../hooks/useAuth';
import Dashboard from '../pages/Dashboard';
import RulesEngine from '../pages/RulesEngine';
import QueueDlq from '../pages/QueueDlq';
import SkuSidebar from '../SkuSidebar';

vi.mock('../hooks/useAuth', () => ({ useAuth: vi.fn() }));
vi.mock('./Charts', () => {
  const Stub = () => null;
  return {
    ConfidenceDoughnutChart: Stub,
    MatchSourcePieChart: Stub,
    VolumeTrendChart: Stub,
    BasicTypeBarChart: Stub,
    RegionDistributionChart: Stub,
    JobsBarChart: Stub,
    RequestVolumeChart: Stub,
  };
});
vi.mock('../api', () => ({
  getDashboardStats: vi.fn(async () => ({})),
  getTagStats: vi.fn(async () => ({})),
  getSystemStatus: vi.fn(async () => ({
    tenant_id: 'default',
    processed_skus: 1234,
    rule_version: 'abc123',
    queue_online: true,
    queue_depth: { realtime: 2, bulk: 5, delayed: 0, inflight: 1, dlq: 1 },
    dlq_size: 1,
  })),
  getRules: vi.fn(async () => []),
  saveRule: vi.fn(),
  reorderRules: vi.fn(),
  testRule: vi.fn(),
  deleteRule: vi.fn(),
  getDeadLetters: vi.fn(async () => [
    {
      task_id: 'task-0001-abcdef',
      job_id: '42',
      batch_id: '42',
      queue: 'matchops:queue:bulk',
      error: 'engine crashed',
      attempts: 3,
      failed_at: '2026-10-26T09:00:00Z',
      payload: {},
    },
  ]),
  replayDeadLetter: vi.fn(async () => ({ job_id: '42', task_id: 'new-task' })),
  login: vi.fn(),
}));

const ROLES = {
  admin: { role: 'admin', isAdmin: true, canEditRules: true, isPlatformAdmin: true },
  rule_engineer: { role: 'rule_engineer', isAdmin: false, canEditRules: true, isPlatformAdmin: false },
  reviewer: { role: 'reviewer', isAdmin: false, canEditRules: false, isPlatformAdmin: false },
};

function renderAs(role, ui) {
  useAuth.mockReturnValue({ me: { username: `${role}_user`, tenant_id: 'default' }, signedIn: true, ...ROLES[role] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SnackbarProvider>{ui}</SnackbarProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => vi.clearAllMocks());

describe('Dashboard system health', () => {
  it('renders System Health above the KPI cards for admins', async () => {
    renderAs('admin', <Dashboard />);
    const health = await screen.findByText('System health');
    const firstKpi = screen.getByText('Avg match confidence');
    // DOCUMENT_POSITION_FOLLOWING: the KPI card comes after the health section
    expect(health.compareDocumentPosition(firstKpi) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(await screen.findByText('1,234')).toBeTruthy();
    expect(screen.getByText('abc123')).toBeTruthy();
  });

  it.each(['reviewer', 'rule_engineer'])('shows only the regular cards for %s', async (role) => {
    renderAs(role, <Dashboard />);
    expect(await screen.findByText('Avg match confidence')).toBeTruthy();
    expect(screen.queryByText('System health')).toBeNull();
    expect(api.getSystemStatus).not.toHaveBeenCalled();
  });
});

describe('RBAC gating', () => {
  it('locks rule creation for reviewers', async () => {
    renderAs('reviewer', <RulesEngine />);
    expect(screen.getByRole('button', { name: 'New rule' }).disabled).toBe(true);
    expect(screen.getByText('Read-only · reviewer')).toBeTruthy();
  });

  it.each(['rule_engineer', 'admin'])('allows rule creation for %s', (role) => {
    renderAs(role, <RulesEngine />);
    expect(screen.getByRole('button', { name: 'New rule' }).disabled).toBe(false);
  });

  it('shows the Queue & DLQ nav item to admins only', () => {
    const { unmount } = renderAs('admin', <SkuSidebar />);
    expect(screen.getByText('Queue & DLQ')).toBeTruthy();
    unmount();
    renderAs('reviewer', <SkuSidebar />);
    expect(screen.queryByText('Queue & DLQ')).toBeNull();
    // A reviewer can still reach the switch-user dialog (e.g. back to admin)
    expect(screen.getByRole('button', { name: 'Switch user or tenant' })).toBeTruthy();
  });
});

describe('DLQ replay', () => {
  it('replays a dead-lettered task after confirmation', async () => {
    renderAs('admin', <QueueDlq />);
    fireEvent.click(await screen.findByRole('button', { name: 'Replay' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Replay' }, { selector: '.MuiDialog-root *' }));
    await waitFor(() => expect(api.replayDeadLetter).toHaveBeenCalled());
    expect(api.replayDeadLetter.mock.calls[0][0]).toBe('task-0001-abcdef');
  });

  it('does not load or offer replay for non-admins', async () => {
    renderAs('reviewer', <QueueDlq />);
    expect(screen.getByText(/dead-letter queue is available to admins/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Replay' })).toBeNull();
    expect(api.getDeadLetters).not.toHaveBeenCalled();
  });
});
