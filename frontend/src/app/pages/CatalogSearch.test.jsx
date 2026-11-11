import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import CatalogSearch from './CatalogSearch';
import * as api from '../api';

vi.mock('../api', () => ({
  searchCatalog: vi.fn(),
  buildCatalogCache: vi.fn(),
  checkCatalogSync: vi.fn(),
  getCatalogSummaryStats: vi.fn(),
}));

function renderCatalogSearch() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SnackbarProvider>
          <CatalogSearch />
        </SnackbarProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('CatalogSearch component', () => {
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

  it('renders search input and domain tabs', async () => {
    renderCatalogSearch();
    expect(screen.getByPlaceholderText(/Search Catalog SKUs/i)).toBeTruthy();
    expect(screen.getByRole('tab', { name: /Market Domain/i })).toBeTruthy();
    expect(screen.getByRole('tab', { name: /Food Domain/i })).toBeTruthy();
  });

  it('displays search results in the catalog table', async () => {
    renderCatalogSearch();
    await waitFor(() => {
      expect(screen.getByText('Organic Whole Milk 1 Gallon')).toBeTruthy();
      expect(screen.getByText('Horizon')).toBeTruthy();
      expect(screen.getByText('Dairy')).toBeTruthy();
    });
  });

  it('filters catalog when typing into the search input', async () => {
    renderCatalogSearch();
    const searchInput = screen.getByPlaceholderText(/Search Catalog SKUs/i);
    fireEvent.change(searchInput, { target: { value: 'Organic' } });

    await waitFor(() => {
      expect(api.searchCatalog).toHaveBeenCalled();
    });
  });
});
