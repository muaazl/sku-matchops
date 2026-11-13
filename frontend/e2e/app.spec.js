import { test, expect } from '@playwright/test';

test.describe('SKU MatchOps SPA Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('http://127.0.0.1:8000/**', async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname === '/health') {
        return route.fulfill({ json: { status: 'ok', database: 'connected' } });
      }
      if (url.pathname === '/jobs/dashboard-stats') {
        return route.fulfill({
          json: {
            kpis: { totalSkus: 0, matchRate: 0, activeJobs: 0, avgSpeedPerMin: 0 },
            throughput: [],
            confidenceBreakdown: [],
            matchSourceDistribution: [],
            domainDistribution: [],
            recentJobs: [],
          },
        });
      }
      if (url.pathname === '/catalog') {
        return route.fulfill({ json: { results: [], total: 0, page: 1, total_pages: 1 } });
      }
      if (url.pathname === '/catalog/summary-stats') {
        return route.fulfill({ json: { catalog: 0, gk: 0, bt: 0, category: 0, brands: 0, bt_gk_map: 0 } });
      }
      if (url.pathname === '/catalog/check-sync') {
        return route.fulfill({ json: { needs_sync: false } });
      }
      if (url.pathname === '/models-status') {
        return route.fulfill({ json: { loaded: true, market: true, food: true } });
      }
      if (url.pathname === '/system/status') {
        return route.fulfill({ json: { queue: { realtime: 0, bulk: 0, dlq: 0 }, workers: [] } });
      }
      if (url.pathname === '/jobs') {
        return route.fulfill({ json: [], headers: { 'x-total-count': '0' } });
      }
      return route.fulfill({ json: {} });
    });
  });

  test('navigates to dashboard and renders header without offline modal', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/#\/dashboard/);
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByText('Server is Offline')).toHaveCount(0);
    await expect(page).toHaveTitle(/SKU MatchOps/i);
  });

  test('navigates to catalog search page and allows typing in search input', async ({ page }) => {
    await page.goto('/#/catalog');
    await expect(page).toHaveURL(/#\/catalog/);
    await expect(page.getByText('Server is Offline')).toHaveCount(0);
    const searchInput = page.locator('input[placeholder*="Search"]').first();
    await expect(searchInput).toBeVisible();
    await searchInput.fill('Anchor Butter');
    await expect(searchInput).toHaveValue('Anchor Butter');
  });

  test('navigates to interactive playground and accepts SKU input', async ({ page }) => {
    await page.goto('/#/interactive');
    await expect(page).toHaveURL(/#\/interactive/);
    await expect(page.getByText('Server is Offline')).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Interactive' })).toBeVisible();
    const skuField = page.locator('input').first();
    await expect(skuField).toBeVisible();
    await skuField.fill('Chicken Kottu');
    await expect(skuField).toHaveValue('Chicken Kottu');
  });
});
