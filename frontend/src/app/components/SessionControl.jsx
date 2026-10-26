import React, { useState } from 'react';
import { Box, Button, Chip, Divider, IconButton, Stack, TextField, Tooltip, Typography } from '@mui/material';
import { useQueryClient } from '@tanstack/react-query';
import { useSnackbar } from 'notistack';
import { KeyRound, LogIn, LogOut, ShieldCheck } from 'lucide-react';
import { login } from '../api';
import { useStore } from '../store';
import { useAuth } from '../hooks/useAuth';
import { IconModal } from './ui';

const ROLE_COLOR = { admin: 'primary', rule_engineer: 'secondary', reviewer: 'default' };
const EMPTY_FORM = { username: '', password: '', apiKey: '' };

// Top-bar session control: a Sign in button when logged out; the role chip and Log out when
// logged in. Logging in unlocks the restricted pages (see RequireLogin).
export default function SessionControl() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const { setSession, clearSession } = useStore();
  const { me, role, signedIn } = useAuth();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [busy, setBusy] = useState(false);

  // Identity changed: never show cached data from the previous session
  const resetScope = () => queryClient.resetQueries();

  const openDialog = () => {
    setForm(EMPTY_FORM);
    setOpen(true);
  };

  const apply = async () => {
    if (!(form.apiKey || (form.username && form.password))) return;
    setBusy(true);
    try {
      if (form.apiKey) {
        setSession({ apiKey: form.apiKey });
      } else if (form.username) {
        const res = await login({ username: form.username, password: form.password });
        setSession({ token: res.access_token });
        enqueueSnackbar(`Signed in as ${form.username} (${res.role})`, { variant: 'success' });
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
    resetScope();
  };

  return (
    <>
      {signedIn ? (
        <Stack direction="row" alignItems="center" spacing={0.5}>
          {role && (
            <Tooltip title={me?.username || ''}>
              <Chip
                size="small"
                variant="outlined"
                color={ROLE_COLOR[role] || 'default'}
                label={role.replace('_', ' ')}
                sx={{ textTransform: 'capitalize', fontWeight: 500 }}
              />
            </Tooltip>
          )}
          <Tooltip title="Log out">
            <IconButton size="small" aria-label="Log out" onClick={signOut}>
              <LogOut size={18} />
            </IconButton>
          </Tooltip>
        </Stack>
      ) : (
        <Button size="small" variant="outlined" startIcon={<LogIn size={16} />} onClick={openDialog}>
          Sign in
        </Button>
      )}

      <IconModal
        open={open}
        onClose={() => setOpen(false)}
        icon={ShieldCheck}
        title="Sign in"
        description="Your role (admin, rule engineer, reviewer) comes from the account you sign in with."
        actions={
          <>
            <Button color="inherit" variant="outlined" fullWidth onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="contained"
              fullWidth
              disabled={busy || !(form.apiKey || (form.username && form.password))}
              onClick={apply}
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
            {/* Hidden submit so Enter signs in */}
            <button type="submit" hidden aria-hidden="true" />
          </Stack>
        </Box>
      </IconModal>
    </>
  );
}
