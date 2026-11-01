import { test, expect } from '@playwright/test';

test.describe('SKU MatchOps SPA Integration Tests', () => {
  test('navigates to dashboard and renders header', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator('body')).toBeVisible();
    await expect(page).toHaveTitle(/SKU MatchOps/i);
  });

  test('navigates to catalog search page and renders search components', async ({ page }) => {
    await page.goto('/catalog');
    await expect(page).toHaveURL(/.*catalog/);
    const searchInput = page.locator('input[placeholder*="Search"], input[type="text"]').first();
    await expect(searchInput).toBeVisible();
  });

  test('navigates to interactive playground and renders SKU input', async ({ page }) => {
    await page.goto('/interactive');
    await expect(page).toHaveURL(/.*interactive/);
    const skuField = page.locator('input').first();
    await expect(skuField).toBeVisible();
  });
});
