import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import { vi } from 'vitest';
import * as api from '../api';
import { useAuth } from '../hooks/useAuth';
import Dashboard from '../pages/Dashboard';
import RulesEngine from '../pages/RulesEngine';
import QueueDlq from '../pages/QueueDlq';
import SkuSidebar from '../SkuSidebar';
import Layout from '../Layout';
import RequireLogin from './RequireLogin';
import { NAV_ITEMS } from '../SkuSidebar';
import { ColorModeProvider } from '../../context/color-context';

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
  getModelsStatus: vi.fn(async () => ({ all_models_ready: true })),
  loadModels: vi.fn(),
}));

const ROLES = {
  admin: { role: 'admin', isAdmin: true, canEditRules: true },
  rule_engineer: { role: 'rule_engineer', isAdmin: false, canEditRules: true },
  reviewer: { role: 'reviewer', isAdmin: false, canEditRules: false },
};
const PUBLIC_PAGES = ['Dashboard', 'Jobs', 'Requests', 'Interactive', 'SKU Results', 'Catalog Search', 'Logs'];

// role 'anonymous' = logged out
function renderAs(role, ui, { path = '/' } = {}) {
  useAuth.mockReturnValue(
    role === 'anonymous'
      ? { me: undefined, signedIn: false, isAdmin: false, canEditRules: false }
      : { me: { username: `${role}_user` }, signedIn: true, ...ROLES[role] }
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <SnackbarProvider>{ui}</SnackbarProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => vi.clearAllMocks());

describe('Dashboard', () => {
  it.each(['admin', 'reviewer', 'rule_engineer'])('has no system health section for %s', async (role) => {
    renderAs(role, <Dashboard />);
    expect(await screen.findByText('Avg match confidence')).toBeTruthy();
    expect(screen.queryByText('System health')).toBeNull();
    expect(api.getSystemStatus).not.toHaveBeenCalled();
  });
});

const renderLayout = (role) =>
  renderAs(
    role,
    <ColorModeProvider initialMode="light">
      <Layout />
    </ColorModeProvider>
  );

describe('Top bar session control', () => {
  it('offers Sign in when logged out', async () => {
    renderLayout('anonymous');
    const header = screen.getByRole('banner');
    expect(within(header).queryByRole('button', { name: 'Log out' })).toBeNull();
    fireEvent.click(within(header).getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByLabelText('Username')).toBeTruthy();
  });

  it.each(['admin', 'reviewer'])('shows the role and a direct Log out for %s', (role) => {
    renderLayout(role);
    const header = screen.getByRole('banner');
    expect(within(header).getByText(role)).toBeTruthy();
    expect(within(header).queryByRole('button', { name: 'Sign in' })).toBeNull();
    expect(within(header).queryByRole('button', { name: /switch user|accessibility/i })).toBeNull();
    fireEvent.click(within(header).getByRole('button', { name: 'Log out' }));
    expect(screen.queryByLabelText('Username')).toBeNull(); // no dialog, logs out immediately
  });
});

describe('Page access', () => {
  const navTitles = () => NAV_ITEMS.map((i) => i.title).filter((t) => screen.queryByText(t));

  it('shows only the public pages when logged out', () => {
    renderAs('anonymous', <SkuSidebar />);
    expect(navTitles()).toEqual(PUBLIC_PAGES);
  });

  it('shows the login-only pages (not Queue) to a logged-in reviewer', () => {
    renderAs('reviewer', <SkuSidebar />);
    expect(navTitles()).toEqual(NAV_ITEMS.map((i) => i.title).filter((t) => t !== 'Queue & DLQ'));
  });

  const guarded = (admin) => (
    <Routes>
      <Route path="/dashboard" element={<div>dashboard page</div>} />
      <Route
        path="/secret"
        element={
          <RequireLogin admin={admin}>
            <div>secret page</div>
          </RequireLogin>
        }
      />
    </Routes>
  );

  it.each([
    ['anonymous', false, 'dashboard page'],
    ['reviewer', false, 'secret page'],
    ['reviewer', true, 'dashboard page'],
    ['admin', true, 'secret page'],
  ])('typing the URL as %s (admin page: %s) lands on the %s', (role, admin, expected) => {
    renderAs(role, guarded(admin), { path: '/secret' });
    expect(screen.getByText(expected)).toBeTruthy();
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
  });
});

describe('Queue & DLQ', () => {
  it('shows live queue counts to admins', async () => {
    renderAs('admin', <QueueDlq />);
    expect(await screen.findByText('5')).toBeTruthy(); // bulk waiting
    expect(screen.getByText('Running now')).toBeTruthy();
  });

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
    expect(api.getSystemStatus).not.toHaveBeenCalled();
  });
});
