import React, { useState } from 'react';
import { Box, Button, Chip, Divider, IconButton, Stack, TextField, Tooltip, Typography } from '@mui/material';
import { useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { KeyRound, ShieldCheck, UserCog } from 'lucide-react';
import { login } from '../api';
import { useStore } from '../store';
import { useAuth } from '../hooks/useAuth';
import { IconModal } from './ui';

const ROLE_COLOR = { admin: 'primary', rule_engineer: 'secondary', reviewer: 'default' };
const EMPTY_FORM = { username: '', password: '', apiKey: '' };

// Compact sidebar-footer session control: shows the server-confirmed role/tenant and opens a
// dialog to sign in as another user (e.g. back to admin after testing as a reviewer), switch
// tenant (platform admins) or sign out.
export default function SessionControl() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const { tenantId, setTenantId, setSession, clearSession } = useStore();
  const { me, role, signedIn, isPlatformAdmin } = useAuth();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [tenantDraft, setTenantDraft] = useState(tenantId);
  const [busy, setBusy] = useState(false);

  // Identity or tenant changed: never show cached data from the previous scope
  const resetScope = () => queryClient.resetQueries();

  const openDialog = () => {
    setTenantDraft(tenantId);
    setForm(EMPTY_FORM);
    setOpen(true);
  };

  const apply = async () => {
    setBusy(true);
    try {
      if (form.apiKey) {
        setSession({ apiKey: form.apiKey });
      } else if (form.username) {
        const res = await login({ username: form.username, password: form.password });
        setSession({ token: res.access_token, tenantId: res.tenant_id });
        enqueueSnackbar(`Signed in as ${form.username} (${res.role})`, { variant: 'success' });
      } else if (isPlatformAdmin) {
        setTenantId(tenantDraft.trim());
      }
      setOpen(false);
      resetScope();
    } catch (e) {
      enqueueSnackbar(e.response?.data?.detail || `Sign-in failed: ${e.message}`, { variant: 'error' });
    } finally {
      setBusy(false);
    }
  };

  const signOut = () => {
    clearSession();
    setOpen(false);
    resetScope();
  };

  return (
    <>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ width: '100%', minWidth: 0 }}>
        {role && (
          <Chip
            size="small"
            variant="outlined"
            color={ROLE_COLOR[role] || 'default'}
            label={role.replace('_', ' ')}
            sx={{ textTransform: 'capitalize', fontWeight: 500 }}
          />
        )}
        <Box sx={{ minWidth: 0, flexGrow: 1 }}>
          <Typography variant="caption" color="text.secondary" noWrap component="div">
            {signedIn ? me?.username : 'Not signed in'}
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap component="div">
            Tenant: {me?.tenant_id || tenantId}
          </Typography>
        </Box>
        <Tooltip title="Switch user or tenant">
          <IconButton size="small" aria-label="Switch user or tenant" onClick={openDialog}>
            <UserCog size={16} />
          </IconButton>
        </Tooltip>
      </Stack>

      <IconModal
        open={open}
        onClose={() => setOpen(false)}
        icon={ShieldCheck}
        title="Switch user or tenant"
        description="Your role (admin, rule engineer, reviewer) and tenant come from the account you sign in with."
        actions={
          <>
            {signedIn ? (
              <Button color="inherit" variant="outlined" fullWidth onClick={signOut}>
                Sign out
              </Button>
            ) : (
              <Button color="inherit" variant="outlined" fullWidth onClick={() => setOpen(false)}>
                Cancel
              </Button>
            )}
            <Button
              variant="contained"
              fullWidth
              disabled={busy || (Boolean(form.username) && !form.password)}
              onClick={apply}
            >
              Apply
            </Button>
          </>
        }
      >
        <Box
          component="form"
          onSubmit={(e) => {
            e.preventDefault();
            apply();
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
            <Tooltip title={isPlatformAdmin ? '' : 'Only platform admins can act on another tenant'}>
              <TextField
                size="small"
                label="Tenant (X-Tenant-ID)"
                value={tenantDraft}
                disabled={!isPlatformAdmin}
                onChange={(e) => setTenantDraft(e.target.value)}
              />
            </Tooltip>
            {/* Hidden submit so Enter applies */}
            <button type="submit" hidden aria-hidden="true" />
          </Stack>
        </Box>
      </IconModal>
    </>
  );
}
