import React, { useEffect, useState } from 'react';
import { Box, Button, Divider, Stack, TextField, Tooltip, Typography } from '@mui/material';
import { useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { LogIn, LogOut, KeyRound, ShieldCheck } from 'lucide-react';
import { login } from '../api';
import { useStore } from '../store';
import { useAuth } from '../hooks/useAuth';
import { IconModal, StatusChip } from './ui';

const ROLE_TONE = { admin: 'ok', rule_engineer: 'warn', reviewer: 'neutral' };

// Top-bar session controls: active tenant (X-Tenant-ID), role badge, sign in / out.
export default function AuthBar() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const { tenantId, setTenantId, setSession, clearSession } = useStore();
  const { me, role, signedIn, isPlatformAdmin } = useAuth();
  const [tenantDraft, setTenantDraft] = useState(tenantId);
  const [loginOpen, setLoginOpen] = useState(false);
  const [form, setForm] = useState({ username: '', password: '', apiKey: '' });
  const [busy, setBusy] = useState(false);

  useEffect(() => setTenantDraft(tenantId), [tenantId]);

  // Identity or tenant changed: never show cached data from the previous scope
  const resetScope = () => queryClient.resetQueries();

  const applyTenant = () => {
    const next = tenantDraft.trim() || 'default';
    if (next !== tenantId) {
      setTenantId(next);
      resetScope();
    }
  };

  const signIn = async () => {
    setBusy(true);
    try {
      if (form.apiKey) {
        setSession({ apiKey: form.apiKey });
      } else {
        const res = await login({ username: form.username, password: form.password });
        setSession({ token: res.access_token, tenantId: res.tenant_id });
        enqueueSnackbar(`Signed in as ${form.username} (${res.role})`, { variant: 'success' });
      }
      setLoginOpen(false);
      setForm({ username: '', password: '', apiKey: '' });
      resetScope();
    } catch (e) {
      enqueueSnackbar(e.response?.data?.detail || `Sign-in failed: ${e.message}`, { variant: 'error' });
    } finally {
      setBusy(false);
    }
  };

  const signOut = () => {
    clearSession();
    resetScope();
  };

  return (
    <Stack direction="row" spacing={1.5} alignItems="center">
      <Tooltip
        title={
          isPlatformAdmin ? 'X-Tenant-ID sent with every request (Enter to apply)' : 'Your tenant is set by your login'
        }
      >
        <TextField
          size="small"
          label="Tenant"
          value={tenantDraft}
          disabled={!isPlatformAdmin}
          onChange={(e) => setTenantDraft(e.target.value)}
          onBlur={applyTenant}
          onKeyDown={(e) => e.key === 'Enter' && applyTenant()}
          inputProps={{ className: 'portal-mono', 'aria-label': 'Active tenant' }}
          sx={{ width: 150 }}
        />
      </Tooltip>

      {role && (
        <Tooltip title={me?.username || ''}>
          <span>
            <StatusChip status={signedIn ? role.replace('_', ' ') : `${role} · dev mode`} tone={ROLE_TONE[role]} />
          </span>
        </Tooltip>
      )}

      {signedIn ? (
        <Button size="small" color="inherit" variant="outlined" startIcon={<LogOut size={14} />} onClick={signOut}>
          Sign out
        </Button>
      ) : (
        <Button size="small" variant="outlined" startIcon={<LogIn size={14} />} onClick={() => setLoginOpen(true)}>
          Sign in
        </Button>
      )}

      <IconModal
        open={loginOpen}
        onClose={() => setLoginOpen(false)}
        icon={ShieldCheck}
        title="Sign in to MatchOps"
        description="Your role (admin, rule engineer, reviewer) and tenant come from your account."
        actions={
          <>
            <Button color="inherit" variant="outlined" fullWidth onClick={() => setLoginOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="contained"
              fullWidth
              disabled={busy || !(form.apiKey || (form.username && form.password))}
              onClick={signIn}
            >
              Sign in
            </Button>
          </>
        }
      >
        <Box
          component="form"
          onSubmit={(e) => {
            e.preventDefault();
            signIn();
          }}
        >
          <Stack spacing={1.5}>
            <TextField
              size="small"
              label="Username"
              autoComplete="username"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
            />
            <TextField
              size="small"
              label="Password"
              type="password"
              autoComplete="current-password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
            />
            <Divider>
              <Typography variant="caption" color="text.secondary">
                or service API key
              </Typography>
            </Divider>
            <TextField
              size="small"
              label="X-API-Key"
              type="password"
              value={form.apiKey}
              onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
              InputProps={{ startAdornment: <KeyRound size={14} style={{ marginRight: 8 }} /> }}
            />
            {/* Hidden submit so Enter signs in */}
            <button type="submit" hidden aria-hidden="true" />
          </Stack>
        </Box>
      </IconModal>
    </Stack>
  );
}
