import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import { useAppStore } from '../../hooks/useAppStore';
import Dashboard from '../../pages/Dashboard';
import * as modelApiService from '../../services/modelApiService';
import Layout from '../layout/Layout';
import SkuSidebar, { NAV_ITEMS } from '../layout/SkuSidebar';
import { CardSkeleton } from './CardSkeleton';
import { ColorModeProvider } from './ColorModeProvider';
import { EmptyState } from './EmptyState';
import { SectionCard } from './SectionCard';
import { StatCard } from './StatCard';
import { TableSkeleton } from './TableSkeleton';

vi.mock('../charts/ConfidenceDoughnutChart', () => ({ ConfidenceDoughnutChart: () => null }));
vi.mock('../charts/MatchSourcePieChart', () => ({ MatchSourcePieChart: () => null }));
vi.mock('../charts/VolumeTrendChart', () => ({ VolumeTrendChart: () => null }));
vi.mock('../charts/BasicTypeBarChart', () => ({ BasicTypeBarChart: () => null }));
vi.mock('../charts/RegionDistributionChart', () => ({ RegionDistributionChart: () => null }));

vi.mock('../../services/jobApiService', () => ({
  getDashboardStats: vi.fn(async () => ({})),
  getTagStats: vi.fn(async () => ({})),
  subscribeJobsStream: vi.fn(() => () => {}),
}));

vi.mock('../../services/modelApiService', () => ({
  getModelsStatus: vi.fn(async () => ({ loaded: true, all_models_ready: true })),
  loadModels: vi.fn(),
}));

function renderWithProviders(ui, { path = '/' } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <SnackbarProvider>{ui}</SnackbarProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  useAppStore.setState({ apiKey: 'dev-local-secret' });
});

describe('Dashboard', () => {
  it('renders dashboard without error', async () => {
    renderWithProviders(<Dashboard />);
    expect(await screen.findByText('Avg match confidence')).toBeTruthy();
    expect(screen.queryByText('System health')).toBeNull();
  });
});

describe('Layout and Navigation', () => {
  it('renders sidebar navigation items', () => {
    renderWithProviders(
      <ColorModeProvider initialMode="light">
        <SkuSidebar />
      </ColorModeProvider>
    );
    for (const item of NAV_ITEMS) {
      expect(screen.getByText(item.title)).toBeTruthy();
    }
  });

  it('renders top bar with model status', async () => {
    modelApiService.getModelsStatus.mockResolvedValueOnce({ loaded: false, loading_in_progress: false });
    renderWithProviders(
      <ColorModeProvider initialMode="light">
        <Layout />
      </ColorModeProvider>
    );
    const header = screen.getByRole('banner');
    expect(await within(header).findByRole('button', { name: 'Load Models' })).toBeTruthy();
  });
});

describe('Design System Primitives', () => {
  it('renders StatCard with value, label, subtext, and trend', () => {
    render(
      <StatCard
        label="Test Metric"
        value="95.5%"
        subtext="Higher is better"
        trend={{ label: '+5%', direction: 'up' }}
      />
    );
    expect(screen.getByText('Test Metric')).toBeTruthy();
    expect(screen.getByText('95.5%')).toBeTruthy();
    expect(screen.getByText('Higher is better')).toBeTruthy();
    expect(screen.getByText('+5%')).toBeTruthy();
  });

  it('renders StatCard with skeletons when isLoading is true', () => {
    const { container } = render(<StatCard label="Test Metric" value="95.5%" subtext="Higher is better" isLoading />);
    expect(screen.queryByText('95.5%')).toBeNull();
    expect(container.querySelectorAll('.MuiSkeleton-root').length).toBeGreaterThan(0);
  });

  it('renders SectionCard with header, action, and content', () => {
    render(
      <SectionCard title="Test Section" subheader="Section description" action={<button>Action</button>}>
        <div>Section Content</div>
      </SectionCard>
    );
    expect(screen.getByText('Test Section')).toBeTruthy();
    expect(screen.getByText('Section description')).toBeTruthy();
    expect(screen.getByText('Section Content')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Action' })).toBeTruthy();
  });

  it('renders EmptyState with title, description, and action button', () => {
    render(
      <EmptyState
        title="No items found"
        description="Try adjusting your filters or search terms."
        action={<button>Reset filters</button>}
      />
    );
    expect(screen.getByText('No items found')).toBeTruthy();
    expect(screen.getByText('Try adjusting your filters or search terms.')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Reset filters' })).toBeTruthy();
  });

  it('renders TableSkeleton and CardSkeleton', () => {
    const { container } = render(
      <div>
        <TableSkeleton rows={3} />
        <CardSkeleton height={100} />
      </div>
    );
    expect(container.querySelectorAll('.MuiSkeleton-root').length).toBeGreaterThanOrEqual(4);
  });
});
