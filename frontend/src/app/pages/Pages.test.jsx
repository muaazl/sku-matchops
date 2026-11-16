import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import { ColorModeProvider } from '../../context/color-context';
import CatalogSearch from './CatalogSearch';
import RulesEngine from './RulesEngine';
import ProcessSKUs from './ProcessSKUs';
import Interactive from './Interactive';
import Jobs from './Jobs';
import SkuResults from './SkuResults';
import * as api from '../api';

vi.mock('../api', () => ({
  searchCatalog: vi.fn(),
  buildCatalogCache: vi.fn(),
  checkCatalogSync: vi.fn(),
  getCatalogSummaryStats: vi.fn(),
  getRules: vi.fn(),
  saveRule: vi.fn(),
  reorderRules: vi.fn(),
  testRule: vi.fn(),
  deleteRule: vi.fn(),
  createBatch: vi.fn(),
  merchantFetch: vi.fn(),
  runInteractiveSingle: vi.fn(),
  getTemplateSuggestions: vi.fn(async () => ({ matched: false })),
  runInteractiveAudit: vi.fn(),
  getJobs: vi.fn(),
  cancelJob: vi.fn(),
  retryJob: vi.fn(),
  subscribeJobStream: vi.fn(() => () => {}),
  getProcessedSkus: vi.fn(),
}));

const MOCK_RULE = {
  rule_id: '10',
  domain: 'food',
  priority: 10,
  description: 'Seafood rule',
  reasoning: 'Tag seafood',
  condition_logic: 'AND',
  is_active: 1,
  conditions: [
    { condition_group: 1, operator: 'AND', condition_type: 'sku_contains', value: 'prawn', negate: 0 },
    { condition_group: 1, operator: 'AND', condition_type: 'gk_contains', value: 'Short Eats', negate: 1 },
  ],
  actions: [{ action_type: 'add_gk', value: 'Seafood' }],
};

function renderPage(ui) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <ColorModeProvider initialMode="dark">
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <SnackbarProvider>{ui}</SnackbarProvider>
        </MemoryRouter>
      </QueryClientProvider>
    </ColorModeProvider>
  );
}

describe('Pages', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.searchCatalog.mockResolvedValue({
      total: 1,
      results: [
        {
          id: 1,
          domain: 'market',
          name: 'Organic Whole Milk 1 Gallon',
          category: 'Dairy',
          brand: 'Horizon',
          flavor: 'Plain',
          basictype: 'milk',
          price: 5.99,
          row_hash: 'hash123',
        },
      ],
    });
    api.getCatalogSummaryStats.mockResolvedValue({
      total_items: 100,
      unique_brands: 20,
    });
  });

  describe('CatalogSearch', () => {
    it('renders search input and domain tabs', async () => {
      renderPage(<CatalogSearch />);
      expect(screen.getByPlaceholderText(/Search Catalog SKUs/i)).toBeTruthy();
      expect(screen.getByRole('tab', { name: /Market Domain/i })).toBeTruthy();
      expect(screen.getByRole('tab', { name: /Food Domain/i })).toBeTruthy();
    });

    it('displays search results in the catalog table', async () => {
      renderPage(<CatalogSearch />);
      await waitFor(() => {
        expect(screen.getByText('Organic Whole Milk 1 Gallon')).toBeTruthy();
        expect(screen.getByText('Horizon')).toBeTruthy();
        expect(screen.getByText('Dairy')).toBeTruthy();
      });
    });

    it('filters catalog when typing into the search input', async () => {
      renderPage(<CatalogSearch />);
      const searchInput = screen.getByPlaceholderText(/Search Catalog SKUs/i);
      fireEvent.change(searchInput, { target: { value: 'Organic' } });

      await waitFor(() => {
        expect(api.searchCatalog).toHaveBeenCalled();
      });
    });
  });

  describe('RulesEngine', () => {
    it('supports per-condition AND/OR operators, NOT toggles, and removes deprecated actions', async () => {
      api.getRules.mockResolvedValue([MOCK_RULE]);
      api.saveRule.mockResolvedValue({ rule_id: '10' });

      renderPage(<RulesEngine />);

      const editBtn = await screen.findByLabelText('Edit rule');
      fireEvent.click(editBtn);

      expect(await screen.findByText('Rule summary')).toBeTruthy();

      const opGroup = screen.getByRole('group', { name: /Condition 2 operator/i });
      expect(within(opGroup).getByRole('button', { name: 'AND' })).toBeTruthy();
      fireEvent.click(within(opGroup).getByRole('button', { name: 'OR' }));

      expect(await screen.findByText(/OR — alternative match/i)).toBeTruthy();

      const actionSelect = screen.getByText('Add Generic Keyword (GK)');
      fireEvent.mouseDown(actionSelect);
      const listbox = await screen.findByRole('listbox');
      expect(within(listbox).getByText('Set Basic Type (BT)')).toBeTruthy();
      expect(within(listbox).queryByText(/set_visibility|Set Visibility/i)).toBeNull();
      expect(within(listbox).queryByText(/normalize_sku|Normalize SKU/i)).toBeNull();

      fireEvent.click(within(listbox).getByText('Set Basic Type (BT)'));
      fireEvent.click(screen.getByRole('button', { name: 'Save rule' }));

      await waitFor(() => {
        expect(api.saveRule).toHaveBeenCalledTimes(1);
      });
      const savedPayload = api.saveRule.mock.calls[0][0];
      expect(savedPayload.conditions[0].operator).toBe('AND');
      expect(savedPayload.conditions[0].condition_group).toBe(1);
      expect(savedPayload.conditions[1].operator).toBe('OR');
      expect(savedPayload.conditions[1].condition_group).toBe(2);
      expect(savedPayload.conditions[1].negate).toBe(1);
      expect(savedPayload.actions[0].action_type).toBe('set_bt');
    });
  });

  describe('ProcessSKUs', () => {
    it('enables portal fetch with Merchant ID and opens token modal when needed', async () => {
      renderPage(<ProcessSKUs />);

      expect(screen.getByRole('heading', { name: 'Process SKUs' })).toBeTruthy();
      const fetchBtn = screen.getByRole('button', { name: /Fetch & Process SKUs/i });
      expect(fetchBtn.disabled).toBe(true);

      const merchantInput = screen.getByLabelText(/Merchant ID/i);
      fireEvent.change(merchantInput, { target: { value: '8842' } });
      expect(fetchBtn.disabled).toBe(false);

      fireEvent.click(fetchBtn);
      expect(await screen.findByText('Enter Bearer Token')).toBeTruthy();
    });
  });

  describe('Interactive', () => {
    it('submits a single SKU and renders results', async () => {
      api.runInteractiveSingle.mockResolvedValue({
        domain: 'food',
        results: [
          {
            matched_catalog_name: 'Red Bull 250ml',
            score: 0.94,
            status: 'confident',
            suggested_bt: 'Energy Drink',
            bt_confidence: 0.95,
            bt_status: 'confident',
            bt_source: 'trained',
            suggested_gk: 'Beverages, Energy Drink',
            gk_confidence: 0.9,
            gk_status: 'confident',
            suggested_region: 'Western',
            region_confidence: 0.88,
            region_status: 'confident',
            rules_applied: [],
          },
        ],
      });

      renderPage(<Interactive />);

      expect(screen.getByRole('heading', { name: 'Interactive' })).toBeTruthy();
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));

      await waitFor(() => {
        expect(api.runInteractiveSingle).toHaveBeenCalledTimes(1);
        expect(api.getTemplateSuggestions).toHaveBeenCalledTimes(1);
      });
      expect(await screen.findByText('Red Bull 250ml')).toBeTruthy();
      expect(screen.getByText(/Basic Type · Energy Drink/i)).toBeTruthy();
      await waitFor(() => {
        expect(screen.queryByText(/Searching for matching template SKU/i)).toBeNull();
      });
    });
  });

  describe('Jobs', () => {
    it('renders jobs table and confirms before cancelling an active job', async () => {
      api.getJobs.mockResolvedValue({
        data: [
          {
            id: 'job_42',
            sheet_name: 'March Menu',
            domain: 'food',
            task: 'pipeline',
            status: 'running',
            total_items: 120,
            high_conf: 95,
            duration_minutes: 1.5,
          },
        ],
        total: 1,
      });
      api.cancelJob.mockResolvedValue({ status: 'cancelling' });

      renderPage(<Jobs />);

      expect(await screen.findByText('job_42')).toBeTruthy();
      expect(screen.getByText('March Menu')).toBeTruthy();

      fireEvent.click(screen.getByRole('button', { name: 'Cancel job' }));
      expect(await screen.findByText('Cancel Job')).toBeTruthy();
      fireEvent.click(screen.getByRole('button', { name: 'Yes' }));

      await waitFor(() => {
        expect(api.cancelJob).toHaveBeenCalledTimes(1);
        expect(api.cancelJob.mock.calls[0][0]).toBe('job_42');
      });
    });
  });

  describe('SkuResults', () => {
    it('lists completed jobs and loads processed SKUs on click', async () => {
      api.getJobs.mockResolvedValue({
        data: [
          {
            id: 'job_99',
            sheet_name: 'Dairy Batch',
            domain: 'market',
            task: 'pipeline',
            status: 'completed',
            total_items: 1,
            high_conf: 1,
          },
        ],
        total: 1,
      });
      api.getProcessedSkus.mockResolvedValue({
        data: [
          {
            id: 'sku_1',
            sku_name: 'Anchor Butter 200g',
            bt: 'Butter',
            gk_json: '["Dairy"]',
            region: 'Dairy & Spreads',
            confidence: 0.96,
          },
        ],
        total: 1,
      });

      renderPage(<SkuResults />);

      const jobCell = await screen.findByText('Dairy Batch');
      fireEvent.click(jobCell);

      await waitFor(() => {
        expect(api.getProcessedSkus).toHaveBeenCalledTimes(1);
      });
      expect(await screen.findByText('Anchor Butter 200g')).toBeTruthy();
      expect(screen.getByText('Butter')).toBeTruthy();
    });
  });
});
