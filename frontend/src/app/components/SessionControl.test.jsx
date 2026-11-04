import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import SessionControl from './SessionControl';
import * as api from '../api';
import { useAuth } from '../hooks/useAuth';
import { useStore } from '../store';

vi.mock('../hooks/useAuth', () => ({ useAuth: vi.fn() }));
vi.mock('../api', () => ({
  login: vi.fn(),
}));

function renderControl() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SnackbarProvider>
          <SessionControl />
        </SnackbarProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('SessionControl component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useStore.setState({
      authToken: '',
      apiKey: '',
      loginModalOpen: false,
      loginPromptDismissed: false,
    });
  });

  it('renders Sign In button when logged out and opens modal on click', async () => {
    useAuth.mockReturnValue({ me: null, role: 'anonymous', signedIn: false });
    renderControl();

    const signInBtn = screen.getByRole('button', { name: /sign in/i });
    expect(signInBtn).toBeTruthy();

    fireEvent.click(signInBtn);
    expect(await screen.findByLabelText(/username/i)).toBeTruthy();
    expect(screen.getByLabelText(/password/i)).toBeTruthy();
  });

  it('submits username and password to log in', async () => {
    useAuth.mockReturnValue({ me: null, role: 'anonymous', signedIn: false });
    api.login.mockResolvedValueOnce({
      access_token: 'valid-jwt-token',
      role: 'admin',
      tenant_id: 'default',
    });

    renderControl();
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    const usernameInput = await screen.findByLabelText(/username/i);
    const passwordInput = screen.getByLabelText(/password/i);

    fireEvent.change(usernameInput, { target: { value: 'admin' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });

    const submitBtn = screen.getByRole('button', { name: /sign in/i, selector: '.MuiDialogActions-root button' });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(api.login).toHaveBeenCalledWith({ username: 'admin', password: 'password123' });
      expect(useStore.getState().authToken).toBe('valid-jwt-token');
    });
  });

  it('shows logged in role chip and log out button when authenticated', () => {
    useAuth.mockReturnValue({
      me: { username: 'test_admin' },
      role: 'admin',
      signedIn: true,
      isAdmin: true,
    });

    renderControl();
    expect(screen.getByText('admin')).toBeTruthy();
    const logoutBtn = screen.getByRole('button', { name: /log out/i });
    expect(logoutBtn).toBeTruthy();

    fireEvent.click(logoutBtn);
    expect(useStore.getState().authToken).toBe('');
  });
});
